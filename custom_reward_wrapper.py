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
        else:
            speed = info.get('speed', 0)
            # Slow down before turns: sharper turn = lower target speed
            if 'track_angle' in info:
                angle_diff = abs(info['track_angle'] - (self.last_angle if self.last_angle is not None else 0))
                # The sharper the turn (higher angle_diff), the lower the allowed speed
                # Example: target_speed = 6 - 10 * angle_diff (tune as needed)
                target_speed = max(1, 6 - 10 * angle_diff)
                if speed > target_speed:
                    reward -= (speed - target_speed) * (2 + 8 * angle_diff)  # Stronger penalty for sharper turns
                else:
                    reward += 1  # Small bonus for being at/below target speed before turn
            else:
                # Fallback: encourage slow movement
                if speed > 1:
                    reward -= (speed - 1) * 2
                else:
                    reward += 1
        if 'track_angle' in info and 'checkpoint' in info:
            if self.last_checkpoint is not None and info['checkpoint'] < self.last_checkpoint:
                reward -= 20  # Large penalty for going backwards
            self.last_checkpoint = info['checkpoint']
            self.last_angle = info['track_angle']
        return obs, reward, terminated, truncated, info
