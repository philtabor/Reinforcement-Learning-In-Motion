import gym

env = gym.make('FrozenLake-v0')

for episode in range(5):
	observation = env.reset()
	done = False
	for step in range(20):
		env.render()
		action = 1 # 0 = left, 1 = down, 2 = right, 3 = up
		print(observation, action)
		observation, reward, done, info = env.step(action)
		if done:
			print('Episode ', episode, 'finished after ', step, 'timesteps')
			break
