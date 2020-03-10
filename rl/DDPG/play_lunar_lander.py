import gym
from ddpg import Agent
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--checkpoint_dir")
args = parser.parse_args()

env = gym.make("LunarLanderContinuous-v2")
# TODO paper uses alpha=1e-4, beta=1e-3
# TODO For Q we included L2 weight decay of 10−2 and used a discount factor of γ = 0.99
# initilized other layers with -1/sqrt(fan-in) -- was mine fan-out?
# TODO OU noise had theta 0.15 and sigma 0.2
agent = Agent(alpha=0.000025, beta=0.00025, state_dim=8, tau=0.001, env=env,
	          batch_size=64, fc1_dim=400, fc2_dim=300, n_actions=2, checkpoint_dir=args.checkpoint_dir)
if args.checkpoint_dir:
	agent.load_models()

observation = env.reset()
for _ in range(10000):
  env.render()
  action = agent.choose_action(observation, noise=False)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()