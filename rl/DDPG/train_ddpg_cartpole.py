
"""
Implementation of DDPG, following 
https://arxiv.org/abs/1509.02971
and
https://www.youtube.com/watch?v=jDll4JSI-xo

ToDo
- define the mu and Q networks
- build a replay buffer
- figure out the noise generator thing, make a class 
- build the training loop

"""
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OrnsteinUhlenbeck(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
    	if self.x0 is not None:
    		self.x_prev = self.x0
    	else:
    		self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ReplayBuffer(object):
	def __init__(self, max_size, input_shape, n_actions):
		self.max_size = max_size
		self.input_shape = input_shape
		self.n_actions = n_actions
		self.state_buf = np.zeros((self.max_size, *input_shape))
		self.next_state_buf = np.zeros((self.max_size, *input_shape))
		self.action_buf = np.zeros((self.max_size, n_actions))
		self.reward_buf = np.zeros((self.max_size, 1))
		self.terminal_buf = np.zeros((self.max_size, 1), dtype=np.float32)
		self.pos = 0

	def store(self, state, action, reward, next_state, done):
		index = self.pos % self.max_size
		self.state_buf[index] = state
		self.next_state_buf[index] = next_state
		self.action_buf[index] = action
		self.reward_buf[index] = reward
		# stores a multiplicative flag to be used against reward to zero invalid rewards.
		self.terminal_buf[index] = 1 - int(done)
		self.pos += 1


	def sample_batch(self, batch_size):
		sample_size = min(self.pos, self.max_size)
		sample_indices = np.random.choice(sample_size, batch_size)

		states = self.state_buf[sample_indices]
		next_states = self.next_state_buf[sample_indices]
		actions = self.action_buf[sample_indices]
		rewards = self.reward_buf[sample_indices]
		terminals = self.terminal_buf[sample_indices]

		return (states, actions, rewards, next_states, terminals)


class Base(nn.Module):
	def __init__(self, name, n_actions, input_dims, fc1_shape, fc2_shape,
			     checkpoint_dir='tmp/ddpg', bn_track_running_stats=False):
		super(Base, self).__init__()
		self.name = name
		self.fc1_shape = fc1_shape
		self.fc2_shape = fc2_shape
		self.checkpoint_file = os.path.join(checkpoint_dir, "ddpg_" + name)
		self.device = T.device('cpu')

		# TODO: track_running_stats = False?  uses batch stats rather than running est during eval.
		self.bn_track_running_stats = bn_track_running_stats

		self.fc1 = nn.Linear(*input_dims, fc1_shape)
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		self.bn1 = nn.BatchNorm1d(fc1_shape, track_running_stats=self.bn_track_running_stats)

		self.fc2 = nn.Linear(fc1_shape, fc2_shape)
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		nn.init.uniform_(self.fc2.bias.data, -f2, f2)
		self.bn2 = nn.BatchNorm1d(fc2_shape, track_running_stats=self.bn_track_running_stats)

	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))


class Actor(Base):
	def __init__(self, name, n_actions, input_dims, fc1_shape, fc2_shape, action_bound, alpha):
		super(Actor, self).__init__(name, n_actions, input_dims, fc1_shape, fc2_shape)
		self.n_actions = n_actions
		self.action_bound = action_bound

		self.mu = nn.Linear(fc2_shape, n_actions)
		f3 = 0.003
		nn.init.Uniform_(self.mu.weight.data, -f3, f3)
		nn.init.Uniform_(self.mu.bias.data, -f3, f3)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)

		self.to(self.device)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		mu = F.tanh(self.mu(x))
		return mu


class Critic(Base):
	def __init__(self, name, n_actions, input_dims, fc1_shape, fc2_shape, action_bound, beta):
		super(Critic, self).__init__(name, n_actions, input_dims, fc1_shape, fc2_shape)
		self.n_actions = n_actions
		self.action_bound = action_bound

		self.action_projection = nn.Linear(n_actions, fc2_shape)
		# TODO no initialization for the action projection? should it have bias?

		f3 = 0.003
		self.q = nn.Linear(fc2_shape, 1)
		nn.init.Uniform_(self.q.weight.data, -f3, f3)
		nn.init.Uniform_(self.q.bias.data, -f3, f3)

		self.optimizer = optim.Adam(self.parameters(), lr=beta)

		self.to(self.device)

	def forward(self, x, actions, q_target):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))

		# TODO - strange double RELU here
		a = F.relu(self.action_projection(actions))
		x = F.relu(T.add(x, a))

		q = self.q(x)

		return q





















