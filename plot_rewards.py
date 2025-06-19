import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from car_racing_new import make_env, load_or_create_model


def collect_rewards(model, episodes=20, steps=10000):
    rewards = []
    for ep in range(episodes):
        env = make_env(render_mode=None)
        obs = env.reset()
        total_reward = 0
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if done.any():
                break
        rewards.append(total_reward)
        env.close()
    return rewards


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.show()


def main():
    env = make_env(render_mode=None)
    # Use the trained model from car_racing_new.py (ppo_carracing.zip)
    model = PPO.load("ppo_carracing.zip", env=env)
    rewards = collect_rewards(model, episodes=20, steps=10000)
    plot_rewards(rewards)

if __name__ == "__main__":
    main()
