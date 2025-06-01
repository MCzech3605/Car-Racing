import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from gym.wrappers import GrayScaleObservation, ResizeObservation

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

if __name__ == "__main__":
    # Create and wrap the environment
    env = gym.make('CarRacing-v2', continuous=True, render_mode="rgb_array")
    env = PreprocessCarRacing(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # Create the PPO agent
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_carracing_tensorboard/")

    # Train the agent
    model.learn(total_timesteps=100_000)

    # Save the model
    model.save("ppo_carracing")

    # Re-create environment for visualization with human render mode
    vis_env = gym.make('CarRacing-v2', continuous=True, render_mode="human")
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
