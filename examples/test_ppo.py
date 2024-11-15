import torch
import gymnasium as gym
import numpy as np
from simple_aircraft_env import SimpleAircraftEnv
from train_ppo import ActorCritic, PPOTrainer
import time
import argparse

def test_policy(model_path, num_episodes=20, render_delay=0.03):
    # Initialize environment with rendering
    env = gym.make('SimpleAircraftEnv-v0', render_mode="human")
    
    # Initialize trainer (for normalization function)
    trainer = PPOTrainer()
    
    # Load model
    checkpoint = torch.load(model_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    
    print("\nTesting model...")
    print(f"Best recorded reward: {checkpoint.get('best_reward', 'N/A')}")
    
    total_reward = 0
    successes = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\nStarting Episode {episode + 1}")
        
        while not done:
            # Normalize and process state
            state_norm = trainer.normalize_observations(state)
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(trainer.device)
            
            # Get action from policy
            with torch.no_grad():
                action_mean, _, _ = trainer.model(state_tensor)
                action = action_mean.cpu().numpy()[0]
            
            # Take action in environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Add delay to make visualization easier to follow
            time.sleep(render_delay)
            
            # Print progress
            if steps % 20 == 0:
                print(f"  Step {steps}, Current Reward: {episode_reward:.2f}", end='\r')
        
        # Episode complete
        print(f"\nEpisode {episode + 1} complete!")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps Taken: {steps}")
        
        total_reward += episode_reward
        if episode_reward > 0:  # Assuming positive reward means success
            successes += 1
    
    # Print final statistics
    print("\nTest Summary:")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    print(f"Success Rate: {successes/num_episodes:.1%}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model.pt", 
                       help="Path to model file")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.03,
                       help="Delay between steps for visualization (seconds)")
    args = parser.parse_args()
    
    # Register environment
    gym.register(
        id='SimpleAircraftEnv-v0',
        entry_point='simple_aircraft_env:SimpleAircraftEnv',
    )
    
    print(f"Testing model: {args.model}")
    test_policy(args.model, args.episodes, args.delay)