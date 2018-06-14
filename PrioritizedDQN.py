import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers, initializers
from collections import deque
import copy
import gym
import numpy as np
import sys
import matplotlib.pyplot as plt


class NN(Chain):
    def __init__(self, n_in, n_out):
        super(NN, self).__init__(
            L1=L.Linear(n_in, 100),
            L2=L.Linear(100, 100),
            L3=L.Linear(100, 100),
            Q_value=L.Linear(100, n_out, initialW=initializers.Normal(scale=0.05))
        )

    def Q_func(self, x):
        h1 = F.leaky_relu(self.L1(x))
        h2 = F.leaky_relu(self.L2(h1))
        h3 = F.leaky_relu(self.L3(h2))
        return F.identity(self.Q_value(h3))


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
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
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.tree_capacity = capacity
        self.tree_real_capacity = 0

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p
        if self.tree_real_capacity < self.tree_capacity:
            self.tree_real_capacity += 1

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 5), dtype=object), np.empty((n, 1), dtype=np.float32)
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class PrioritizedDQN(object):
    def __init__(self, n_st, n_act, seed=0):
        super(PrioritizedDQN, self).__init__()
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_st = n_st
        self.n_act = n_act
        self.model = NN(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.memory_size = 2**13  # 8192
        self.memory = Memory(self.memory_size)
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.001
        self.epsilon_min = 0
        self.exploration = 1000
        self.target_update_freq = 30

    def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.store((st, act, r, st_dash, ep_end))

    def forward(self, st, act, r, st_dash, ep_end, ISWeights):
        s = Variable(st)
        s_dash = Variable(st_dash)
        Q = self.model.Q_func(s)
        Q_dash = self.model.Q_func(s_dash)
        Q_dash_target = self.target_model.Q_func(s_dash).data
        Q_dash_idmax = np.asanyarray(list(map(np.argmax, Q_dash.data)))
        max_Q_dash = np.asanyarray([Q_dash_target[i][Q_dash_idmax[i]] for i in range(len(Q_dash_idmax))])
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in range(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        squared_error = F.squared_error(Q, Variable(target))
        loss = F.mean(squared_error * ISWeights)
        td_error = F.add(Q, Variable(-1 * target))
        self.loss = loss.data
        return loss, td_error

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in range(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        idx, batch, ISWeights = self.memory.sample(self.batch_size)
        st, act, r, st_dash, ep_end = self.parse_batch(batch)

        self.model.cleargrads()
        loss, td_error = self.forward(st, act, r, st_dash, ep_end, ISWeights)

        loss.backward()
        self.optimizer.update()

        abs_errors = np.sum(np.abs(td_error.data), axis=1)
        self.memory.batch_update(idx, abs_errors)

    def get_action(self, st):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_act), 0
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8), max(Q)

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        memory_real_size = self.memory.tree_real_capacity
        if memory_real_size >= self.memory_size:
            self.experience_replay()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
                self.reduce_epsilon()
        self.step += 1

    def save_model(self, outputfile):
        serializers.save_npz(outputfile, self.model)

    def load_model(self, inputfile):
        serializers.load_npz(inputfile, self.model)


if __name__ == "__main__":
    env_name = "CartPole-v0"
    seed = 0
    env = gym.make(env_name)
    view_path = 'video/' + env_name

    n_st = env.observation_space.shape[0]

    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = np.arange(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = PrioritizedDQN(n_st, n_act, seed)
    # env.Monitor.start(view_path, video_callable=None, force=True, seed=seed)

    list_t = []
    list_loss = []
    for i_episode in range(3000):
        print("episode_num" + str(i_episode))
        observation = env.reset()
        for t in range(200):
            env.render()
            state = observation.astype(np.float32).reshape((1, n_st))
            act_i = agent.get_action(state)[0]
            action = action_list[act_i]
            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32).reshape((1, n_st))

            reward_true = t / 200

            agent.stock_experience(state, act_i, reward_true, state_dash, ep_end)
            agent.train()
            if ep_end:
                print('max t:', t)
                print('loss:', agent.loss)
                list_t.append(t)
                list_loss.append(agent.loss)
                break
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(list_t)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(list_loss)
    plt.show()
    # env.Monitor.close()

    agent.save_model('PrioritizedDQN.model')
