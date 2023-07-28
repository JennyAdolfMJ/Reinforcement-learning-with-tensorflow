import tensorflow as tf
import numpy as np

class DQN_Brain:
    def __init__(self, 
                 action_space,
                 observation_space, 
                 batch_size=32, 
                 replace_target_iter=300,
                 memory_size=500,
                 e_greedy=0.9,
                 e_greedy_increment=None,
                 reward_decay=0.9):
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.memory = np.zeros((self.memory_size, observation_space * 2 + 2))
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.gamma = reward_decay

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.build_network()
    
    def build_network(self):
        self.q_eval = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(None, self.observation_space), activation='relu'),
        tf.keras.layers.Dense(self.action_space)
        ])
        self.q_eval.compile(optimizer='rmsprop', loss='mse')

        self.q_target = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(None, self.observation_space), activation='relu'),
        tf.keras.layers.Dense(self.action_space)
        ])
    
    def store_transaction(self, observation, action, reward, observation_):
        transaction = np.hstack((observation, [action, reward], observation_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transaction
        # print(transaction)

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_values = self.q_eval.predict(observation)
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.action_space)
        return action
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            for idx in range(len(self.q_eval.layers)):
                self.q_target.get_layer(index=idx).set_weights(self.q_eval.get_layer(index=idx).get_weights())

        arange = np.min([self.memory_size, self.memory_counter])
        sample_index = np.random.choice(arange, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_eval = self.q_eval.predict(batch_memory[:, :self.observation_space], batch_size=self.batch_size)
        q_next = self.q_target.predict(batch_memory[:, -self.observation_space:], batch_size=self.batch_size)

        q_target = q_eval.copy()
        act_col = batch_memory[:, self.observation_space].astype(int)
        reward_col = batch_memory[:, self.observation_space+1]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, act_col] = reward_col + self.gamma * np.max(q_next, axis=1)

        self.q_eval.fit(batch_memory[:, :self.observation_space], q_target)

        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        self.learn_step_counter += 1
    
    def learn_DQ(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            for idx in range(len(self.q_eval.layers)):
                self.q_target.get_layer(index=idx).set_weights(self.q_eval.get_layer(index=idx).get_weights())

        arange = np.min([self.memory_size, self.memory_counter])
        sample_index = np.random.choice(arange, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.q_target.predict(batch_memory[:, -self.observation_space:], batch_size=self.batch_size)
        q_eval_next = self.q_eval.predict(batch_memory[:, -self.observation_space:], batch_size=self.batch_size)

        q_eval = self.q_eval.predict(batch_memory[:, :self.observation_space], batch_size=self.batch_size)

        q_target = q_eval.copy()
        act_col = batch_memory[:, self.observation_space].astype(int)
        reward_col = batch_memory[:, self.observation_space+1]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        temp = q_next[batch_index, np.argmax(q_eval_next, axis=1)]
        q_target[batch_index, act_col] = reward_col + self.gamma * temp

        self.q_eval.fit(batch_memory[:, :self.observation_space], q_target)

        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        self.learn_step_counter += 1
