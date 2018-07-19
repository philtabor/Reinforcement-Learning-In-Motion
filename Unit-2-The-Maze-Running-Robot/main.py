import numpy as np
from environment import Maze
from agent import Agent
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    maze = Maze() 
    robot = Agent(maze, alpha=0.1, randomFactor=0.25)
    moveHistory = []  
    for i in range(5000):               
        if i % 1000 == 0:
            print(i)        
        while not maze.isGameOver():
            state, _ = maze.getStateAndReward()
            action = robot.chooseAction(state, maze.allowedStates[state])
            maze.updateMaze(action)
            state, reward = maze.getStateAndReward()
            robot.updateStateHistory(state, reward)
        robot.learn()                
        moveHistory.append(maze.steps)
        maze = Maze() 

    maze = Maze()
    robot = Agent(maze, alpha=0.99, randomFactor=0.25)     
    moveHistory2 = []    
    for i in range(5000):        
        if i % 1000 == 0:
            print(i)
        while not maze.isGameOver():
            state, _ = maze.getStateAndReward()
            action = robot.chooseAction(state, maze.allowedStates[state])
            maze.updateMaze(action)
            state, reward = maze.getStateAndReward()
            robot.updateStateHistory(state, reward)    
        robot.learn()       
        moveHistory2.append(maze.steps)
        maze = Maze()  
    plt.semilogy(moveHistory, 'b--', moveHistory2, 'r--')    
    plt.legend(['alpha=0.1','alpha=0.99'])
    plt.show()

