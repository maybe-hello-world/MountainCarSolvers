import gym
import tensorflow as tf
import numpy as np
import logging
from typing import Union

from tqdm import tqdm
from collections import defaultdict

reward_shaper = defaultdict(lambda: lambda obs, next_obs, rew: rew)  # default reward shaping is no shaping
reward_shaper.update({
    # reward shaping for MountainCar-v0: maximize speed
    "MountainCarContinuous-v0":
        lambda obs, next_obs, rew:
        abs(1 + next_obs[0] - obs[0]) ** 5
        + (10 + obs[0] * 30 if obs[0] > 0.3 else rew)
        + (obs[0] * 10000 if obs[0] > 0.42 else rew)
})


class ContinuousSolver:
    def __init__(
            self,
            gym_env: str,
            actor_lr: float = 0.001,
            critic_lr: float = 0.001,
            gamma: float = 0.95
    ):
        """
        Initiate an agent with given parameters

        :param gym_env: Gym env name (should be continuous)
        :param actor_lr: learning rate for actor's optimizer
        :param critic_lr: learning rate for critic's optimizer
        :param gamma: discount rate
        """
        self.env_name = gym_env
        self.env = gym.make(gym_env)

        self.observation_n = sum(self.env.observation_space.shape)
        self.action_n = sum(self.env.action_space.shape)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        tf.reset_default_graph()
        self.obs = tf.placeholder(tf.float32, [self.observation_n], "state")
        self.action = tf.placeholder(dtype=tf.float32, name="action")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        self._create_actor()
        self._create_critic()

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def _create_actor(self) -> None:
        """
        Create actor network (****ing continuous actionspace...) and save optimizer training operation

        :return:
        """
        with tf.variable_scope("actor"):

            self.actor_network = tf.layers.Dense(32, activation=tf.nn.relu)(tf.expand_dims(self.obs, 0))
            self.mu = tf.layers.Dense(1)(self.actor_network)
            self.mu = tf.squeeze(self.mu)
            self.mu = tf.tanh(self.mu)  # -1 .. 1 for MouCar

            self.sigma = tf.layers.Dense(1)(self.actor_network)
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 0.001

            self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist.sample()
            self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])

            # Loss and train op
            self.actor_loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.actor_loss -= 0.01 * self.normal_dist.entropy()

            self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
            self.actor_train_op = self.actor_optimizer.minimize(
                self.actor_loss, global_step=tf.contrib.framework.get_global_step())

    def _create_critic(self) -> None:
        """
        Create critic implementation

        :return: None
        """
        with tf.variable_scope("critic"):

            self.critic_network = tf.layers.Dense(32, activation=tf.nn.relu)(tf.expand_dims(self.obs, 0))
            self.value = tf.layers.Dense(1)(self.critic_network)
            self.value = tf.squeeze(self.value)
            self.critic_loss = tf.squared_difference(self.value, self.target)
            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
            self.critic_train_op = self.critic_optimizer.minimize(
                self.critic_loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        """
        Predict action(s) using normal distribution and agents output

        :param obs: observation from the environment
        :return: sampled action or array of actions
        """
        return self.sess.run(self.action, {self.obs: obs})

    def update(self, obs: np.ndarray, action: float, reward: float, next_obs: np.ndarray) -> None:
        """
        Update actor and critic with played step

        :param obs: observation at step i
        :param action: action(s) taken at step i
        :param reward: reward received for action on step i
        :param next_obs: observation at step i+1
        :return: None, because experience is priceless :)
        """
        # predict next value from critic
        value_next = self.sess.run(self.value, {self.obs: next_obs})
        td_target = reward + self.gamma * value_next
        td_error = td_target - self.sess.run(self.value, {self.obs: obs})

        # update value estimator
        feed_dict = {self.obs: obs, self.target: td_target}
        _, loss = self.sess.run([self.critic_train_op, self.critic_loss], feed_dict)

        # update policy estimator
        feed_dict = {self.obs: obs, self.target: td_error, self.action: action}
        _, loss = self.sess.run([self.actor_train_op, self.actor_loss], feed_dict)

        return

    def train(self, episodes: int) -> None:
        """
        Perform training for given amount of episodes (games)
        :param episodes: number of episodes to be played
        :return: Not really anything useless
        """

        logging.info("Training begin")

        rewards = []
        shaped_rewards = []
        rew_shaping = reward_shaper[self.env_name]

        for i in tqdm(range(episodes)):
            done = False
            true_rew = 0.0
            shaped_rew = 0.0
            obs = self.env.reset()

            while not done:
                action = self.predict(obs)
                next_obs, reward, done, _ = self.env.step(np.array([action]))

                true_rew += reward
                reward = rew_shaping(obs, next_obs, reward) / 10
                shaped_rew += reward

                self.update(obs, action, reward, next_obs)
                obs = next_obs

            rewards.append(true_rew)
            shaped_rewards.append(shaped_rew)
            print(f"True reward: {true_rew}, modified: {shaped_rew}, episode num: {i}")
        logging.info("Training ended")

    def show(self) -> None:
        """
        Show must go on!
        Display agent's play with current policy
        :return: None except your pleasure
        """

        obs = self.env.reset()
        while True:
            act = self.predict(obs)
            obs, _, done, _ = self.env.step(np.array([act]))
            obs = obs if not done else self.env.reset()
            self.env.render()


if __name__ == '__main__':
    agent = ContinuousSolver("MountainCarContinuous-v0")
    agent.train(100)
    agent.show()
