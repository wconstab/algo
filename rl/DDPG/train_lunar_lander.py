from ddpg import Agent
import gym
import numpy as np

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_shape=8, tau=0.001, env=env,
	          batch_size=64, l1=400, l2=300, n_actions=2)

np.random.seed(0)

score_history = []
for i in range(1000):
	done = False
	score = 0
	state = env.reset()
	while not done:
		action = agent.choose_action(state)
		new_state, reward, done, info = env.step(action)
		agent.remember(state, action, reward, new_state, int(done))
		agent.learn()
		score += reward
		state = new_state

	score_history.append(score)
	print('Episode {i} score {score}, 100-game avg {avg}'.format(i=i,
		                                                         score=score,
		                                                         avg=np.mean(score_history[-100:])))

	if i % 25 == 0:
		agent.save_models()

