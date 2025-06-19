import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym.wrappers import GrayScaleObservation, ResizeObservation
from car_racing_new import make_env

# Preprocessing wrapper for CarRacing
def make_preprocessed_env(render_mode="human"):
    env = gym.make('CarRacing-v2', continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env

def main():
    env = make_env(render_mode="rgb_array")
    # Only load the model, do not archive or create a new one
    model = PPO.load("ppo_carracing", env=env)
    from car_racing_new import record_video_of_agent
    record_video_of_agent(model, video_dir="videos", steps=10_000)

if __name__ == "__main__":
    main()
