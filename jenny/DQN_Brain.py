import tensorflow as tf
import numpy as np


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# self-implementation of a dueling layer
class Dueling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.value = tf.keras.layers.Dense(10, activation='relu')
        self.advantage = tf.keras.layers.Dense(10, activation='relu')
        self.add = tf.keras.layers.Add()
        self.subtract = tf.keras.layers.Subtract()

    def call(self, inputs):
        advantage = self.subtract([self.advantage(inputs), tf.reduce_mean(self.advantage(inputs), axis=1, keepdims=True)])
        res = self.add([self.value(inputs), advantage])

        return res


class DQN_Brain:
    def __init__(self, 
                 action_space,
                 observation_space, 
                 batch_size=32, 
                 replace_target_iter=300,
                 memory_size=500,
                 e_greedy=0.9,
                 e_greedy_increment=None,
                 reward_decay=0.9,
                 prioritized=False,
                 dueling=False):
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.gamma = reward_decay
        self.prioritized = prioritized
        self.dueling = dueling
        self.lose_weight = np.ones((self.batch_size,1))

        if prioritized:
            self.memory = Memory(memory_size)
        else:
            self.memory = np.zeros((self.memory_size, observation_space * 2 + 2))

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.build_network()
    
    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, input_shape=(None, self.observation_space), activation='relu'))
        if self.dueling:
            model.add(Dueling())
        model.add(tf.keras.layers.Dense(self.action_space))

        return model


    def build_network(self):
        # Evaluation network
        self.q_eval = self.build_model()
        self.q_eval.compile(optimizer='rmsprop', loss='mse')

        # Target network
        self.q_target = self.build_model()

    
    def store_transaction(self, observation, action, reward, observation_):
        transaction = np.hstack((observation, [action, reward], observation_))

        if self.prioritized:
            self.memory.store(transaction)
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transaction

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

        if self.prioritized:
            sample_index, batch_memory, self.lose_weight  = self.memory.sample(self.batch_size)
        else:
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

        if self.prioritized:
            errors = tf.reduce_sum(tf.abs(q_target - q_eval), axis=1)
            self.q_eval.fit(batch_memory[:, :self.observation_space], q_target, sample_weight=self.lose_weight)
            self.memory.batch_update(sample_index, errors)
        else:
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
