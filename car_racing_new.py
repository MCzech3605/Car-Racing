import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from gym.wrappers import GrayScaleObservation, ResizeObservation
import os
import shutil
from datetime import datetime

# Preprocessing wrapper for CarRacing
class PreprocessCarRacing(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = GrayScaleObservation(env, keep_dim=True)
        self.env = ResizeObservation(self.env, (84, 84))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_angle = None
        self.last_checkpoint = None
        self.last_on_grass = False

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_angle = 0
        self.last_checkpoint = 0
        self.last_on_grass = False
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Custom reward shaping
        # 1. Punish for entering grass
        if 'on_grass' in info and info['on_grass']:
            reward -= 10  # Large penalty for grass
        # 2. Prevent reward for collecting checkpoints behind
        if 'track_angle' in info and 'checkpoint' in info:
            if self.last_checkpoint is not None and info['checkpoint'] < self.last_checkpoint:
                reward -= 20  # Large penalty for going backwards
            self.last_checkpoint = info['checkpoint']
            self.last_angle = info['track_angle']
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Create and wrap the environment
    env = gym.make('CarRacing-v2', continuous=True, render_mode="rgb_array")
    env = CustomRewardWrapper(env)
    env = PreprocessCarRacing(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # Load the PPO agent from the zip file if it exists, otherwise create a new one
    # Directory to archive loaded zips
    archive_dir = "ppo_carracing_archive"
    os.makedirs(archive_dir, exist_ok=True)

    # Move and/or rename the zip after loading
    if os.path.exists("ppo_carracing.zip"):
        model = PPO.load("ppo_carracing", env=env, tensorboard_log="./ppo_carracing_tensorboard/")
        print("Loaded model from ppo_carracing.zip. Continuing training...")
        # Prepare archive path
        archive_path = os.path.join(archive_dir, "ppo_carracing.zip")
        if os.path.exists(archive_path):
            # If already exists, rename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(archive_dir, f"ppo_carracing_{timestamp}.zip")
        shutil.move("ppo_carracing.zip", archive_path)
        print(f"Moved loaded zip to: {archive_path}")
    else:
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_carracing_tensorboard/")
        print("No existing model found. Starting new training...")

    # Continue training the agent
    model.learn(total_timesteps=1)

    # Save the model
    model.save("ppo_carracing")

    # Re-create environment for visualization with human render mode
    vis_env = gym.make('CarRacing-v2', continuous=True, render_mode="human")
    vis_env = CustomRewardWrapper(vis_env)
    vis_env = PreprocessCarRacing(vis_env)
    vis_env = DummyVecEnv([lambda: vis_env])
    vis_env = VecFrameStack(vis_env, n_stack=4)

    # Evaluate the agent visually
    obs = vis_env.reset()
    total_reward = 0
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vis_env.step(action)
        total_reward += reward[0]
        vis_env.render()
        if done.any():
            break
    print(f"Total reward: {total_reward}")
    vis_env.close()
