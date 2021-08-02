import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from ReinforcementModel import LinearNetWork, Trainer
from helper import plot

###############################################################################################
###############################################################################################
#######################           Global variables       ######################################

MAX_MEMORY = 100_000
batch_size = 1000
learning_rate = 0.001

###############################################################################################
###############################################################################################
###############################################################################################
###################          Creation the agent              ##################################


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate for billman equation 
        self.memory = deque(maxlen=MAX_MEMORY) #list pop from left if we reach the max lenght of the memory
        self.model = LinearNetWork(11, 256, 3)
        self.trainer = Trainer(self.model, learning_rate, gamma=self.gamma)
    
    def getPoints(self,head):
        left = Point(head.x - 20, head.y)
        right = Point(head.x + 20, head.y)
        up = Point(head.x, head.y - 20)
        down = Point(head.x, head.y + 20)
        
        return right,left,up,down 

###############################################################################################
###############################################################################################
###############################################################################################
###################          Checking the current direction                 ###################

    def checkDirections(self,game):
        dirLeft = game.direction == Direction.LEFT
        dirRight = game.direction == Direction.RIGHT
        dirUp = game.direction == Direction.UP
        dirDown = game.direction == Direction.DOWN

        return dirLeft,dirRight,dirUp,dirDown

    ######   Checking the collision if exists in all possible block next to the head of the snake  #####
    
    def checkCollisions(self,game,directions,points):

            #current Direction 

            dirLeft,dirRight,dirUp,dirDown=directions

            #points next to the head of the snake 

            right,left,up,down=points
            
            # Checking for the 3 possible risks 

            straight_risk=(dirRight and game.is_collision(right)) or (dirLeft and game.is_collision(left)) or (dirUp and game.is_collision(up)) or (dirDown and game.is_collision(down))

            
            right_risk=(dirUp and game.is_collision(right) )or (dirDown and game.is_collision(left)) or (dirLeft and game.is_collision(up)) or (dirRight and game.is_collision(down))

         
            left_risk=(dirDown and game.is_collision(right)) or (dirUp and game.is_collision(left)) or (dirRight and game.is_collision(up)) or (dirLeft and game.is_collision(down))       

            return straight_risk, right_risk , left_risk 

    def get_state(self, game):


        head = game.snake[0]
        points=self.getPoints(head)
        directions=self.checkDirections(game)
        
        straight_risk, right_risk , left_risk =self.checkCollisions(game,directions,points)

        state = [   straight_risk,right_risk,left_risk,
                    
                    #directions=(dirLeft,dirRight,dirUp,dirDown)
                    directions[0],
                    directions[1],
                    directions[2],
                    directions[3],
            
                    # Food location 
                    game.food.x < game.head.x,  # food located on the left
                    game.food.x > game.head.x,  # food located on the  right
                    game.food.y < game.head.y,  # food located up 
                    game.food.y > game.head.y   # food located down
            ]

        return np.array(state, dtype=int)

    #saving games

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 


###############################################################################################
###############################################################################################
###############################################################################################
###################          Training                                       ###################
    

    def train_long_memory(self):
        if len(self.memory) > batch_size:
            # each minisample is a list of tuples with lenth =batch_size 
            mini_sample = random.sample(self.memory, batch_size) 
        else:
            #else we train all existing games 
            mini_sample = self.memory
   
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    


    def get_action(self, state,threshold):
        #Generaion of random moves only if the number of games is less than a threshold 
        # and a random value between  and 200 when it is less than epsilon 
        # and this means when sometimes we get the action from the model 

        self.epsilon = threshold - self.n_games

        # action= [straight,Right,Left]

        action = [0,0,0]
        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)
            action[move] = 1
        else:
            # Models of torch get the prediction with the state as an input 
            # Syntax pred=model(state)

            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # to get the index of the max value in the last linear layer 
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    threshold=70
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get the action
        action = agent.get_action(state_old,threshold)

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # Saving the game
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()