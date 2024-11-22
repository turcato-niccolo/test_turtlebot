import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 32)
		self.l2 = nn.Linear(32, 32)
		self.l3 = nn.Linear(32, 32)
		self.l4 = nn.Linear(32, 32)
		self.l5 = nn.Linear(32, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		a = F.relu(self.l4(a))
		return self.max_action * torch.tanh(self.l5(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 32)
		self.l2 = nn.Linear(32, 32)
		self.l3 = nn.Linear(32, 32)
		self.l4 = nn.Linear(32, 32)
		self.l5 = nn.Linear(32, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		q = F.relu(self.l3(q))
		q = F.relu(self.l4(q))
		return self.l5(q)

class DDPG(object):
	def __init__(
			self,
			state_dim,
			action_dim,
			max_action,
			discount=0.99,
			tau=0.005,
			policy_freq=2,
			 policy_noise=0.2,
			 noise_clip=0.5,
			 OVER=2.,
			 UNDER=1.,
			*args, **kargs):
		self.max_action = max_action
		self.noise_clip = noise_clip
		self.policy_noise = policy_noise
		self.total_it = 0
		self.policy_freq = policy_freq
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.OVER = OVER
		self.UNDER = UNDER


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		# Select action according to policy and add clipped noise
		noise = (
				torch.randn_like(action) * self.policy_noise
		).clamp(-self.noise_clip, self.noise_clip)

		next_action = (
				self.actor_target(next_state) + noise
		).clamp(-self.max_action, self.max_action)
		target_Q = self.critic_target(next_state, next_action)
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# EXPECTILE LOSS
		UNDER = self.UNDER
		OVER = self.OVER
		UNDER, OVER = UNDER / max(UNDER, OVER), OVER / max(UNDER, OVER)
		residual = current_Q - target_Q
		under_estimations = residual * (residual < 0).to(torch.float32)
		over_estimations = residual * (residual >= 0).to(torch.float32)
		critic_loss = (
			under_estimations**2 * UNDER + over_estimations**2 * OVER
		).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic(state, self.actor(state)).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
	
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
		self.actor_target = copy.deepcopy(self.actor)

	# def load(self, filename):
	# 	self.critic.load_state_dict(torch.load(filename + "_critic"))
	# 	self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
	# 	self.critic_target = copy.deepcopy(self.critic)

	# 	self.actor.load_state_dict(torch.load(filename + "_actor"))
	# 	self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
	# 	self.actor_target = copy.deepcopy(self.actor)
		