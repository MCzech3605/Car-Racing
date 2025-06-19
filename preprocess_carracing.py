import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation

class PreprocessCarRacing(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = GrayScaleObservation(env, keep_dim=True)
        self.env = ResizeObservation(self.env, (84, 84))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
