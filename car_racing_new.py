import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import os
import shutil
from datetime import datetime
from gym.wrappers import RecordVideo
from preprocess_carracing import PreprocessCarRacing
from custom_reward_wrapper import CustomRewardWrapper


def make_env(render_mode="rgb_array"):
    env = gym.make('CarRacing-v2', continuous=True, render_mode=render_mode)
    env = CustomRewardWrapper(env)
    env = PreprocessCarRacing(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


def archive_model_zip(zip_name="ppo_carracing.zip", archive_dir="ppo_carracing_archive"):
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, zip_name)
    if os.path.exists(zip_name):
        if os.path.exists(archive_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(archive_dir, f"ppo_carracing_{timestamp}.zip")
        shutil.move(zip_name, archive_path)
        print(f"Moved loaded zip to: {archive_path}")


def load_or_create_model(env, tensorboard_log="./ppo_carracing_tensorboard/"):
    if os.path.exists("ppo_carracing.zip"):
        model = PPO.load("ppo_carracing", env=env, tensorboard_log=tensorboard_log)
        print("Loaded model from ppo_carracing.zip. Continuing training...")
        archive_model_zip()
    else:
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        print("No existing model found. Starting new training...")
    return model


def evaluate_model(model, render_mode="human", steps=10_000):
    env = make_env(render_mode=render_mode)
    obs = env.reset()
    total_reward = 0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        env.render()
        if done.any():
            break
    print(f"Total reward: {total_reward}")
    env.close()


def record_video_of_agent(model, video_dir="videos", steps=10_000):
    import gym
    from gym.wrappers import RecordVideo
    os.makedirs(video_dir, exist_ok=True)
    # Wrap RecordVideo first, then apply custom wrappers
    base_env = gym.make('CarRacing-v2', continuous=True, render_mode="rgb_array")
    record_env = RecordVideo(base_env, video_folder=video_dir, episode_trigger=lambda x: True)
    env = CustomRewardWrapper(record_env)
    env = PreprocessCarRacing(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done.any():
            break
    env.close()
    print(f"Video saved to {video_dir}")


def main():
    env = make_env(render_mode="rgb_array")
    model = load_or_create_model(env)
    model.learn(total_timesteps=1)
    model.save("ppo_carracing")
    # evaluate_model(model, render_mode="human", steps=10_000)
    record_video_of_agent(model, video_dir="videos", steps=10_000)


if __name__ == "__main__":
    main()
