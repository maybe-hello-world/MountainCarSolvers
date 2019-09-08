import gym
import numpy as np

env = gym.make('MountainCarContinuous-v0')
obs = env.reset()
done = False


def decide(observation):
    position, velocity = observation
    lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
             0.3 * (position + 0.9) ** 4 - 0.008)
    ub = -0.07 * (position + 0.38) ** 2 + 0.07
    if lb < velocity < ub:
        action = 0.99  # push right
    else:
        action = -0.99  # push left
    return action


try:
    while True:
        env.render()
        act = decide(obs)
        obs, reward, done, info = env.step(np.array([act]))
        if done:
            obs = env.reset()
            print(reward)
except KeyboardInterrupt:
    env.close()
    raise
