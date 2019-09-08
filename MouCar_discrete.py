import gym
import tensorflow as tf
import numpy as np
import logging
import itertools

from tqdm import tqdm
from typing import Tuple, Iterable, Union
from collections import deque, defaultdict

reward_shaper = defaultdict(lambda: lambda obs, next_obs, rew: rew)  # default reward shaping is no shaping
reward_shaper.update({
    # reward shaping for MountainCar-v0: maximize speed
    "MountainCar-v0":
        lambda obs, next_obs, rew: abs(1 + next_obs[0] - obs[0]) ** 5 + (10 + obs[0] * 30 if obs[0] > 0.3 else rew)
})


class DiscreteSolver:
    """
    Vanilla DQN agent for discrete environments
    """
    def __init__(
            self,
            gym_env: str,
            replay_batch_size: int = 100,
            learning_rate: float = 0.001,
            gamma: float = 0.99,
            copy_every: int = 500
    ) -> None:
        """
        Initiate the agent with given parameters

        :param gym_env: Name of the Gym environment
        :param replay_batch_size: batch_size to be used by replay (and warmup counter also)
        :param learning_rate: learning rate of the agent
        :param gamma: discount factor for future rewards
        :param copy_every: how often to copy q_network to target_network
        """
        self.env_name = gym_env
        self.env = gym.make(gym_env)
        self.replay_buffer = deque([], maxlen=10000)

        self.observation_n = sum(self.env.observation_space.shape)
        self.lr = learning_rate
        self.gamma = gamma
        self.replay_bs = replay_batch_size
        self.copy_every = copy_every

        self.epsilon_generator = lambda x: (max(1 - i / x, 0.01) for i in itertools.count())
        self.q_network = self._build_network(self.observation_n)
        self.target_network = self._build_network(self.observation_n)

    def _sample_from_buffer(self) -> Iterable[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Returns random memories from replay buffer to learn"""
        idxs = np.floor(np.random.random_sample(size=self.replay_bs) * len(self.replay_buffer))
        return (self.replay_buffer[i] for i in idxs.astype(int))

    def _build_network(self, input_dim: int) -> tf.keras.Sequential:
        """Build model for predicting"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=input_dim, activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        model.single_predict = lambda obs: model.predict(np.expand_dims(obs, axis=0))[0]  # tired of infinite [0]s
        return model

    def _update_weights(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """
        Returns index of action to be performed by the agent.
        Uses epsilon-greedy policy and Q values for choosing actions.

        :param observation: observation from an environment
        :return: index of action to be performed
        """

        if np.random.random_sample() < next(self.epsilon_generator):
            return np.random.randint(0, self.env.action_space.n)
        q_values = self.q_network.single_predict(observation)
        return np.argmax(q_values)

    def __imagine_future(self, datatuple):
        """ Function for transforming replay memory to data for keras.fit"""
        _obs, _act, _rew, _nextobs, _done = datatuple
        if not _done:
            # imagine future reward
            future_q_values = self.q_network.single_predict(_nextobs)
            optimal_q = self.gamma * np.amax(future_q_values)
            _rew += optimal_q

        target_q_values = self.target_network.single_predict(_obs)
        target_q_values[_act] = _rew  # change predicted to real
        return np.concatenate([_obs, target_q_values])

    def replay(self) -> None:
        """
        Replay batch_size memories from replay buffer and train on them to be stronger :)

        :return: Nothing valuable, because experience is priceless
        """
        samples = self._sample_from_buffer()
        samples = np.array(list(map(self.__imagine_future, samples)))
        x = samples[:, :self.observation_n]
        y = samples[:, self.observation_n:]
        self.q_network.fit(x, y, epochs=1, verbose=0)

    def train(self, steps: int) -> None:
        """
        Train agent during given amount of steps

        :param steps: amount of steps to be played
        :return Nothing
        """

        self.epsilon_generator = self.epsilon_generator(int(steps // 2))

        logging.info("Training begin")
        obs = self.env.reset()
        steps += self.replay_bs * 2  # for replay buffer
        episode_sum = 0.0
        shaped_episode_sum = 0.0
        episode_num = 0
        for i in tqdm(range(steps), desc="Training"):
            action = self.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            true_reward = reward
            reward = reward_shaper[self.env_name](obs, next_obs, reward)
            self.replay_buffer.append((obs, action, reward, next_obs, done))
            episode_sum += true_reward
            shaped_episode_sum += reward
            obs = next_obs
            if done:
                obs = self.env.reset()
                print(f"True reward: {episode_sum}, modified: {shaped_episode_sum}, episode num: {episode_num}")
                episode_sum = 0.0
                shaped_episode_sum = 0.0
                episode_num += 1
            if i > self.replay_bs * 2:
                self.replay()
            if i % self.copy_every == 0:
                self._update_weights()
        logging.info("Training complete")

    def show(self) -> None:
        """
        Visualize training
        :return: Meh
        """
        obs = self.env.reset()
        while True:
            act = self.act(obs)
            obs, _, done, _ = self.env.step(act)
            obs = obs if not done else self.env.reset()
            self.env.render()


if __name__ == '__main__':
    agent = DiscreteSolver("MountainCar-v0", replay_batch_size=10)
    agent.train(100000)
    agent.show()
