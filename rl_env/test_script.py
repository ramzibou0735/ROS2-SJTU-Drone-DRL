import gymnasium as gym
from gymnasium.envs.registration import register
import rclpy

# Import environment directly
from drone_env import SecurityDroneEnv

# Register with correct path


def test_environment():
    env = None
    try:
        # Create environment instance
        env = SecurityDroneEnv()  # Direct instantiation
        print("Environment created successfully")
        episodes = 1000
        steps_per_episode = 1000

        for episode in range(episodes):
            observation = env.reset()
            print("Environment reset")
            episode_reward = 0
            
            for step in range(steps_per_episode):
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
                    
            print(f"Episode {episode + 1} reward: {episode_reward}")

    except Exception as e:
        print(f"Test failed: {e}")
        rclpy.shutdown()
    finally:
        if env is not None:
            env.close()
        else:
            print("Environment instance not found")

if __name__ == "__main__":
    test_environment()