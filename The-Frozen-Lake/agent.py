import numpy as np

class Agent(object):
	def __init__(self, gamma, states, policy):
		self.gamma = gamma # discount parameter		
		self.policy = policy # mapping of states to actions		
		self.memory = [] # state reward pairs
		self.statesReturns = {} # states and the discounted returns that followed
		self.v = self.initV(states)
		
	def initV(self, states):
		V = {}
		for state in states:
			V[state] = 0
		return V

	def updateMemory(self, state, reward):
		self.memory.append((state,reward))	

	def updateV(self):
		G = 0	
		# assemble discounted future rewards from the agent's memory	
		for state, reward in reversed(self.memory):			
			if state not in self.statesReturns:
				self.statesReturns[state] = [G]
			else:
				self.statesReturns[state].append(G)
			G = reward + self.gamma * G

		# use discounted future rewards to calculate averages for each state
		for state in self.statesReturns:
			self.v[state] = np.mean(self.statesReturns[state])
		
		self.memory = []

	def chooseAction(self, state):		
		action = self.policy[state]		
		return action
		
	def printV(self):
		for state in self.v:
			print(state, '%.5f' % self.v[state])
