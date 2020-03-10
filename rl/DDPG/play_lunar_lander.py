import gym
from ddpg import Agent

env = gym.make("LunarLanderContinuous-v2")
agent = Agent(alpha=0.000025, beta=0.00025, state_dim=8, tau=0.001, env=env,
	          batch_size=64, fc1_dim=400, fc2_dim=300, n_actions=2)
agent.load_models()

observation = env.reset()
for _ in range(10000):
  env.render()
  action = agent.choose_action(observation, noise=False)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()