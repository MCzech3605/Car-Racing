import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym.wrappers import GrayScaleObservation, ResizeObservation

# Preprocessing wrapper for CarRacing
def make_preprocessed_env(render_mode="human"):
    env = gym.make('CarRacing-v2', continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env

if __name__ == "__main__":
    # Create environment for visualization
    env = make_preprocessed_env(render_mode="human")

    # Load the latest model
    model = PPO.load("ppo_carracing", env=env)

    # Evaluate the agent visually
    obs = env.reset()
    total_reward = 0
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        env.render()
        if done.any():
            break
    print(f"Total reward: {total_reward}")
    env.close()
