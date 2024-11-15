import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from time import time
from torch.distributions import Normal
from simple_aircraft_env import SimpleAircraftEnv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 128  # Increased network size
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # Initialize for smaller initial actions
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        action_mean = self.actor_mean(shared)
        action_std = self.actor_log_std.exp()
        value = self.critic(shared)
        return action_mean, action_std, value

class PPOTrainer:
    def __init__(self):
        # Register and create environment
        gym.register(
            id='SimpleAircraftEnv-v0',
            entry_point='simple_aircraft_env:SimpleAircraftEnv',
        )
        self.env = gym.make('SimpleAircraftEnv-v0')
        self.eval_env = gym.make('SimpleAircraftEnv-v0')
        
        # Initialize dimensions and device
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model and optimizer
        self.model = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        # PPO hyperparameters
        self.rollout_length = 2048
        self.batch_size = 64
        self.num_epochs = 10
        self.clip_param = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Initialize tracking variables
        self.timesteps = 0
        self.rewards_history = []
        self.episode_length_history = []
        self.success_rate_history = []
        self.value_loss_history = []
        self.policy_loss_history = []
        self.entropy_history = []
        self.steps_history = []
        
        print(f"Initialized PPO trainer on device: {self.device}")
        print(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
        print("Hyperparameters:")
        print(f"  Rollout length: {self.rollout_length}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Number of epochs: {self.num_epochs}")
        print(f"  Learning rate: 3e-4")

    def normalize_observations(self, obs):
        # Fixed normalization values based on environment constants
        BOUNDARY_SIZE = 400.0
        SENSOR_RANGE = 200.0
        
        # Normalize observations to [-1, 1] range
        obs = np.array(obs, dtype=np.float32)
        obs[0:2] /= BOUNDARY_SIZE  # Position
        obs[3:6] /= SENSOR_RANGE   # Sensor readings
        obs[6:8] /= BOUNDARY_SIZE  # Relative waypoint position
        return obs

    def collect_rollout(self):
        states, actions, rewards, values = [], [], [], []
        log_probs, dones = [], []
        
        state, _ = self.env.reset()
        episode_rewards = []
        current_episode_reward = 0
        episodes_completed = 0
        
        print("\nCollecting rollout...")
        
        for step in range(self.rollout_length):
            if step % 100 == 0:
                print(f"  Step {step}/{self.rollout_length}, Episodes completed: {episodes_completed}, "
                    f"Current episode reward: {current_episode_reward:.2f}")
            
            # Normalize state and convert to tensor
            state_norm = self.normalize_observations(state)
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_mean, action_std, value = self.model(state_tensor)
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            
            states.append(state_norm)
            actions.append(action[0].cpu().numpy())
            rewards.append(reward)
            values.append(value.cpu().numpy()[0][0])
            log_probs.append(log_prob.cpu().numpy()[0])
            dones.append(done)
            
            current_episode_reward += reward
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                episodes_completed += 1
                next_state, _ = self.env.reset()
            
            state = next_state
        
        print(f"\nRollout collection complete:")
        print(f"  Episodes completed: {episodes_completed}")
        if episode_rewards:
            print(f"  Average episode reward: {np.mean(episode_rewards):.2f}")
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, values, log_probs, dones, episode_rewards

    def compute_advantages(self, rewards, values, dones):
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)
        
        last_gae = 0
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards)-1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
            returns[t] = last_gae + values[t]
            advantages[t] = last_gae
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        print("\nUpdating policy...")
        
        # Prepare batches
        batch_indices = np.arange(len(states))
        n_batches = len(states) // self.batch_size
        
        for epoch in range(self.num_epochs):
            np.random.shuffle(batch_indices)
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_ids = batch_indices[batch_start:batch_end]
                
                batch_states = states[batch_ids]
                batch_actions = actions[batch_ids]
                batch_old_log_probs = old_log_probs[batch_ids]
                batch_returns = returns[batch_ids]
                batch_advantages = advantages[batch_ids]
                
                # Forward pass
                action_mean, action_std, values = self.model(batch_states)
                dist = Normal(action_mean, action_std)
                
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().mean()
                
                # Calculate policy loss
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()
                
                # Combined loss
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    - self.entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
            
            avg_policy_loss = epoch_policy_loss / n_batches
            avg_value_loss = epoch_value_loss / n_batches
            avg_entropy = epoch_entropy / n_batches
            
            print(f"  Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"    Policy Loss: {avg_policy_loss:.4f}")
            print(f"    Value Loss: {avg_value_loss:.4f}")
            print(f"    Entropy: {avg_entropy:.4f}")
            
            total_policy_loss += avg_policy_loss
            total_value_loss += avg_value_loss
            total_entropy += avg_entropy
        
        return (
            total_policy_loss / self.num_epochs,
            total_value_loss / self.num_epochs,
            total_entropy / self.num_epochs
        )

    def evaluate(self, num_episodes=5):
        total_reward = 0
        episode_lengths = []
        successes = 0
        
        for episode in range(num_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                state_norm = self.normalize_observations(state)
                state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_mean, _, _ = self.model(state_tensor)
                    action = action_mean.cpu().numpy()[0]
                
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
            episode_lengths.append(steps)
            if episode_reward > 0:  # Positive reward indicates success
                successes += 1
        
        return (
            total_reward / num_episodes,
            np.mean(episode_lengths),
            successes / num_episodes
        )
    def plot_training_progress(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards with rolling mean
        if len(self.rewards_history) > 0:
            steps = np.array(self.steps_history)
            rewards = np.array(self.rewards_history)
            
            ax1.plot(steps, rewards, 'b.', alpha=0.3, label='Rewards')
            if len(rewards) >= 10:
                rolling_mean = np.convolve(rewards, np.ones(10)/10, mode='valid')
                ax1.plot(steps[9:], rolling_mean, 'r-', label='Rolling Mean')
            ax1.set_title('Average Reward')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Reward')
            ax1.legend()
        
        # Plot success rate
        if len(self.success_rate_history) > 0:
            ax2.plot(self.steps_history[:len(self.success_rate_history)], 
                    self.success_rate_history, 'g-')
            ax2.set_title('Success Rate')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Success Rate')
        
        # Plot losses
        if len(self.value_loss_history) > 0:
            ax3.plot(self.steps_history, self.value_loss_history, 'b-', label='Value Loss')
            ax3.plot(self.steps_history, self.policy_loss_history, 'r-', label='Policy Loss')
            ax3.set_title('Losses')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Loss')
            ax3.legend()
        
        # Plot episode lengths
        if len(self.episode_length_history) > 0:
            ax4.plot(self.steps_history[:len(self.episode_length_history)], 
                    self.episode_length_history, 'm-')
            ax4.set_title('Average Episode Length')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

    def train(self, total_timesteps=1_000_000, eval_freq=5000):
        print(f"\nStarting training for {total_timesteps} timesteps")
        print(f"Will evaluate every {eval_freq} timesteps")
        best_reward = float('-inf')
        
        try:
            while self.timesteps < total_timesteps:
                print(f"\nTimestep {self.timesteps}/{total_timesteps}")
                
                # Collect experience
                states, actions, rewards, values, log_probs, dones, episode_rewards = self.collect_rollout()
                self.timesteps += self.rollout_length
                
                # Compute returns and advantages
                returns, advantages = self.compute_advantages(rewards, values, dones)
                
                # Update policy
                policy_loss, value_loss, entropy = self.update_policy(
                    states, actions, log_probs, returns, advantages
                )
                
                # Record history
                self.policy_loss_history.append(policy_loss)
                self.value_loss_history.append(value_loss)
                self.entropy_history.append(entropy)
                self.steps_history.append(self.timesteps)
                
                if len(episode_rewards) > 0:
                    self.rewards_history.append(np.mean(episode_rewards))
                
                # Evaluate and plot
                if self.timesteps % eval_freq < self.rollout_length:
                    eval_reward, eval_length, success_rate = self.evaluate()
                    
                    print("\nEvaluation Results:")
                    print(f"  Timestep: {self.timesteps}/{total_timesteps}")
                    print(f"  Eval Reward: {eval_reward:.2f}")
                    print(f"  Success Rate: {success_rate:.2%}")
                    print(f"  Average Episode Length: {eval_length:.1f}")
                    print("-" * 50)
                    
                    self.episode_length_history.append(eval_length)
                    self.success_rate_history.append(success_rate)
                    
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        self.save_model("best_model.pt")
                        print(f"  New best model saved with reward: {best_reward:.2f}")
                
                self.plot_training_progress()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
            self.save_model("interrupted_model.pt")
            print("Saved interrupted model.")
            raise

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps': self.timesteps,
            'best_reward': max(self.rewards_history) if self.rewards_history else float('-inf'),
            'rewards_history': self.rewards_history,
            'success_rate_history': self.success_rate_history,
            'steps_history': self.steps_history
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.timesteps = checkpoint['timesteps']
        self.rewards_history = checkpoint.get('rewards_history', [])
        self.success_rate_history = checkpoint.get('success_rate_history', [])
        self.steps_history = checkpoint.get('steps_history', [])
        return checkpoint.get('best_reward', float('-inf'))

if __name__ == "__main__":
    trainer = PPOTrainer()
    try:
        trainer.train(total_timesteps=1_000_000, eval_freq=5000)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        trainer.save_model("interrupted_model.pt")
        print("Model saved. Exiting...")