import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32 #How many sets to update parameters once
LR = 0.01 # learning rate
EPSILON = 0.9 # greedy policy
GAMMA = 0.9 # reward discount
TARGET_REPLACE_ITER = 100 # target update frequency
MEMORY_CAPACITY = 2000 #Similar to Q-learning, a table: records all [State][action][reward], etc.

env = gym.make('CartPole-v0')
env = env.unwrapped


N_ACTIONS = env.action_space.n #The dimension of the action vector
N_STATES = env.observation_space.shape[0] #The dimension of the state vector


ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape # to confirm the shape


class Net(nn.Module): #A network
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50) #hidden layer
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(50, N_ACTIONS) #output layer
        self.out.weight.data.normal_(0, 0.1) # initialization

        The parameter x of #forward is the state vector, which can be understood as the observation vector
    def forward(self, x): #Call net(x) to automatically call the forward() function
        
        x = self.fc1(x)
        x = F.relu(x) #Input --> hidden layer --> Relu activation function --> output layer
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net() #Build two neural networks, one is eval_net (pioneer) and the other is target_net (predicted target network)

        self.learn_step_counter = 0 # for target updating
        self.memory_counter = 0 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)#optimizer
        self.loss_func = nn.MSELoss() #mean squared error

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON: # greedy #If the random number is less than epsilon, then choose the most greedy action
            actions_value = self.eval_net.forward(x) #Note that the vanguard network is used to predict: so that the parameters can be dynamically adjusted
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) # return the argmax index
        else: # random # Else, choose the action randomly
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    
def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) #Store the current obs, reward and other information into a vector
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #If the total number of learning reaches the size of a batch, update the weight of the vanguard network to the target network
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) #Choose an index from MEMORY to learn from the eval network
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a) # shape (batch, 1)
                                                    # eval_net(b_s) outputs 32 rows of action values ​​corresponding to each b_s by evaluating the network
                                                    # gather(1, b_a) means to aggregate the Q value extraction of each row corresponding to the index b_a
                                                    # q_next does not transfer the error in the reverse direction, so detach; q_next means to output a series of action values ​​corresponding to each b_s_ of 32 lines through the target network
        q_next = self.target_net(b_s_).detach() # detach from graph, don't backpropagate
        # Neural network training may sometimes want to keep some network parameters unchanged,
        # Adjust only some of the parameters.
        # Or train part of the branch network,
        # Do not let its gradient affect the gradient of the main network.
        # At this time, we need to use the detach() function to cut off the backpropagation of some branches.
        # return a new tensor,
        # Detach from the current computation graph.
        # But it still points to the storage location of the original variable,
        # The only difference is that require_grad is false.
        # The obtained tensir never needs a calculator gradient,
        # does not have grad.
        
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # shape (batch, 1)
        # error function
        loss = self.loss_func(q_eval, q_target)
        # optimizer gradient clear
        self.optimizer.zero_grad()
        # Error backpropagation, calculate parameter update value
        loss.backward()
        # Update all parameters of the evaluation network
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, truncated, infos = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2)) #round rounding function with two decimal places

        if done:
            break
        s = s_