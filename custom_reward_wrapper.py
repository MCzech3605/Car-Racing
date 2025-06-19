import gym

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
        if 'on_grass' in info and info['on_grass']:
            reward -= 10  # Large penalty for grass
        if 'track_angle' in info and 'checkpoint' in info:
            if self.last_checkpoint is not None and info['checkpoint'] < self.last_checkpoint:
                reward -= 20  # Large penalty for going backwards
            self.last_checkpoint = info['checkpoint']
            self.last_angle = info['track_angle']
        return obs, reward, terminated, truncated, info
