# Group Assignment of Group *Polaris*

> This is a document about our project and its learning path.
>
> The following is our group's project, and I will attach all the exploration from the beginning to the final assignment.

​	

> ​    Group Name:  Polaris(from South China University of Technology)
> ​	Group Members:
>
> Yongjie Liang (Ragan): Environment&Algorithm selection and code implementation,documentation
>
> Weiyue Zhang: Configuration of the code environment and finding code example
>
> Jieyi Sun:Team leader: task assignment and group management
>
> Ruoqi Chen: Algorithm material data search,ppt production, assist in code implementation
>
> Yuxuan Chen: Assist with running routines ,finding games ,searching information
>
> Yihao Shen:Data processing and visualization(pandas,matplotlib)

## Project1: Exploration of gym and pettingzoo api

> This project is aimed to let us get familiar with the gym's api and have a basic understanding of reinforcement learning

```python
import gym #openAI gym's library
import numpy as np #Use the numpy library for vector and matrix operations

env= gym.make('CartPole-v0') #Select the training game

done = False # done = True means the game is over
cnt = 0 # Indicates how many times the action is done

observation = env.reset() #observation vector stores information about the environment
                          #In this game observation includes: the position, speed, etc. of the pole
while not done:
     env.render() #Render the display screen
    
     cnt +=1 #Indicates how many operations the agent has performed
    
     action = env.action_space.sample() #Randomly sample each time in the action space to let the agent operate
    
     observation, reward, done, _ = env.step(action) #Return this operation (that is, the result of interacting with the environment)

     if done: #if the game is over (the pole fell)
         break # exit the loop

print('game last %d'%cnt,' move') #Output how many times the agent has operated
```

The output is just like below:

```python
game last 15  move
game last 12  move
game last 12  move
game last 43  move
game last 15  move
game last 26  move
game last 22  move
game last 24  move
game last 14  move
game last 16  move
game last 81  move
game last 15  move
game last 55  move
game last 17  move
game last 34  move
game last 18  move
game last 18  move
game last 29  move
game last 18  move
game last 49  move
```



After that, based on my knowledge of reinforcement learning, I wrote a simple memory optimization for the agent, which is **EasyOptimise.py** in the folderm.The source code is as follows



```python
import gym #openAI gym's library
import numpy as np #Use the numpy library for vector and matrix operations
from gym import wrappers # This library can help us convert the video of the game into audio format

env= gym.make('CartPole-v0') #Select the training game

bestlength = 0 #The best case is stored -> the most lasting episode
episode_length = [] #Stored is the number of continuous steps for each episode

best_weight = np.zeros(4) #4-dimensional array: It stores the configuration of each parameter in the best case (weight?)

for i in range(100): #Do 100 trainings and play 100 games per training (the next for loop)
    
    new_weight = np.random.uniform(-1.0,1.0,4) #The parameters (weights?) are initialized according to a uniform distribution each time
    
    length = [] #length is used to record the result of this time
    
    for j in range(100):
        #Initialize environment parameters
        
        observation = env.reset()
        done = False
        cnt = 0
        
        while not done: #one game
            #env.render() #Comment this out because 10000 renders are too time consuming
        
            cnt += 1
            #The meaning of the following sentence is: if the dot product of the parameter vector (weight) and the environment parameter vector is greater than 0, then take action[1], otherwise take action[2]
            action =1 if np.dot(observation,new_weight) >0 else 0
            #Enter the action calculated above through the step function
            observation, reward, done, _ = env.step(action)
            
            if done: #If the game is over, then exit this 100-step loop
                break
    
        length.append(cnt)
        average_length = float(sum(length)/len(length))
        if average_length > bestlength:
            bestlength = average_length
            best_weight = new_weight
        episode_length.append(average_length)
        #if i%20 ==0:
        print(average_length)
        print('\tbest length is',bestlength)

done = False
cnt=0
observation = env.reset()

while not done:
    cnt +=1
    action = 1 if np.dot(observation,best_weight)>0 else 0
    observation, reward, done, _ = env.step(action)
    
    if done:
        break
    
        
print('with the best weight, game last %d'%cnt,' move') #Output how many times the agent has operated
```

with this EasyOptimise , the move can easily attained  to 200:

```python
9.0
	best length is 9.0
8.5
	best length is 9.0
8.666666666666666
	best length is 9.0
9.0
	best length is 9.0
9.2
	best length is 9.2
9.333333333333334
	best length is 9.333333333333334
9.428571428571429
	best length is 9.428571428571429
9.5
	best length is 9.5
9.555555555555555
	best length is 9.555555555555555
9.4
	best length is 9.555555555555555
9.272727272727273
	best length is 9.555555555555555
9.25
	best length is 9.555555555555555
9.23076923076923
	best length is 9.555555555555555
9.142857142857142
	best length is 9.555555555555555
9.2
	best length is 9.555555555555555
9.25
	best length is 9.555555555555555
9.235294117647058
	best length is 9.555555555555555
9.222222222222221
	best length is 9.555555555555555
9.263157894736842
	best length is 9.555555555555555
...
	best length is 72.91935483870968
16.11111111111111
	best length is 72.91935483870968
16.035714285714285
	best length is 72.91935483870968
16.06896551724138
	best length is 72.91935483870968
16.4
	best length is 72.91935483870968
    
...


69.69565217391305
	best length is 200.0
69.48936170212765
	best length is 200.0
69.39583333333333
	best length is 200.0
69.53061224489795
	best length is 200.0
69.34
	best length is 200.0
69.0
	best length is 200.0
69.34615384615384
	best length is 200.0
69.09433962264151
	best length is 200.0
69.20370370370371
	best length is 200.0
69.2
	best length is 200.0
69.01785714285714
	best length is 200.0
68.84210526315789
	best length is 200.0
68.5
	best length is 200.0
68.27118644067797
	best length is 200.0
68.8
	best length is 200.0
69.06557377049181
	best length is 200.0
69.06451612903226
	best length is 200.0
68.93650793650794
	best length is 200.0
68.6875
	best length is 200.0
68.72307692307692
	best length is 200.0
68.6969696969697
	best length is 200.0
68.83582089552239
	best length is 200.0
69.08823529411765
	best length is 200.0
68.81159420289855
	best length is 200.0
68.75714285714285
	best length is 200.0
68.91549295774648
	best length is 200.0
68.72222222222223
	best length is 200.0
68.67123287671232
	best length is 200.0
68.47297297297297
	best length is 200.0
68.36
	best length is 200.0
68.23684210526316
	best length is 200.0
68.07792207792208
	best length is 200.0
68.35897435897436
	best length is 200.0
68.34177215189874
	best length is 200.0
68.3
	best length is 200.0
68.17283950617283
	best length is 200.0
68.42682926829268
	best length is 200.0
68.40963855421687
	best length is 200.0
68.44047619047619
	best length is 200.0
68.43529411764706
	best length is 200.0
68.79069767441861
	best length is 200.0
68.67816091954023
	best length is 200.0
68.5340909090909
	best length is 200.0
68.71910112359551
	best length is 200.0
68.74444444444444
	best length is 200.0
68.65934065934066
	best length is 200.0
68.55434782608695
	best length is 200.0
68.53763440860214
	best length is 200.0
68.41489361702128
	best length is 200.0
68.52631578947368
	best length is 200.0
68.39583333333333
	best length is 200.0
68.52577319587628
	best length is 200.0
68.73469387755102
	best length is 200.0
68.66666666666667
	best length is 200.0
68.46
	best length is 200.0
with the best weight, game last 65  move

```

Although it is not a RL algorithms strictly, this is a good example to  show  how computers remember the experience and it is a good stepping-stone to RL.

## Project2 :Q-learning

The source code is **Qlearning.ipynb**

```python
import gym #openAI gym's library
import numpy as np #Use the numpy library for vector and matrix operations
from gym import wrappers # This library can help us convert the video of the game into audio format
import matplotlib.pyplot as plt #visualization

env= gym.make('CartPole-v0') #Select the training game
MAXSTATE = 10**4 #possibille discrete state
GAMMA = 0.9 #parameters in the Q-learning mathematics:
ALPHA = 0.01 #parameters in the Q-learning mathematics: learning rate

def max_dict(d): #iterate a dictionary and return the max_key and its value
                                          #Goal: to find the action expecting the maxium reward for a given state
    max_v = float('-inf') #inf:infinity ∞
   
    for key,val in d.items():
        if val >max_v:
            max_v = val
            max_key = key
    return max_key,max_v

def create_bins(): #how to break a continous space into discrete space (because the game is continous)
    bins = np.zeros((4,10)) # 4 parameters in the observation
    The #linspace function can return an array (the array here refers to the ndarray array) within the specified range (start to stop), which contains num evenly spaced samples.
    bins[0] = np.linspace(-4.8,4.8,10) # cart position
    bins[1] = np.linspace(-5,5,10) # cart velocity
    bins[2] = np.linspace(-0.418,0.418,10) # pole angle
    bins[3] = np.linspace(-5,5,10) # pole velocity
    return bins

def assign_bins(observation,bins): #Given a specific state and what bin will it fall into (continious->digital)
    state = np.zeros(4) #state is a four-dimensional vector, and each vector marks the subscript of the bin to which the state falls
    for i in range(4):
        state[i] = np.digitize(observation[i],bins[i]) #Which of the 10 buckets does the parameter fall into? (Think of a picture of a normally distributed ball) that returns an array index that falls on the pin
    return state

def get_state_as_string(state): # Stringify each state. Is it convenient for dictionary encoding? In fact, I don't know where the convenience is - Liang Yongjie
    #print(state[0],'\n')
    #print(state[1],'\n')
    #print(state[2],'\n')
    #print(state[3],'\n')
    string_state = ''.join(str(int(e)) for e in state)
    return string_state

def get_all_states_as_string(): #This function generates an integer-to-string sequence of [0000,9999] for state encoding
    states = []
    for i in range(MAXSTATE):
 
        states.append(str(i).zfill(4)) #The fill function is built into the str class for high-order zero-filling
    return states

def initialize_Q():
    Q = {} #declare here that Q is a dict type (dictionary)
                                                           #Q can be understood as a mapping table between action-state and expected reward
    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {} #Each Q[state] is also a dictionary
        for action in range(env.action_space.n): #For each state, each action taken is recorded
            Q[state][action] = 0 #This is not a two-dimensional array, but: For Q, index with key=state, for Q[state] index with [action]
    return Q

def play_one_game(bins , Q , eps = 0.5):
    observation = env.reset()
    done = False
    cnt =0 #number of moves in an episode
    state = get_state_as_string(assign_bins(observation,bins))
    total_reward = 0
                                                            #epsilon greedy algorithm
    while not done:
        cnt = 1
        if np.random.uniform() < eps: #If the random number p (0~1) is less than ε, then randomly sample in the action space
            act = env.action_space.sample()
        else: #If the random number p (0~1) is greater than ε, then only select the action with the largest profit in the current state
            act = max_dict(Q[state])[0]
                                                            # Over time, the best action will be chosen more and more frequently, because choosing it will get more benefits.
        observation, reward, done, _ = env.step(act) #Interact the action extracted by the greedy algorithm with the environment
        
        total_reward += reward #Record the reward in the total reward
        
        if done and cnt < 200: #Ununderstood: Since the maximum reward is 200, a reward of -300 will be disastrous as a fallover penalty
            reward = -300
            
        state_new = get_state_as_string(assign_bins(observation,bins))#encode to statecode the observation obtained by just interacting with the environment
        
        al,max_q_slal = max_dict(Q[state_new]) #Find the optimal action in the current state and the corresponding future prediction reward value
        Q[state][act] += ALPHA*(reward+GAMMA*max_q_slal - Q[state][act])
        state,act = state_new,al
    return total_reward, cnt
        
        
def play_many_games(bins,N=10000):
    Q = initialize_Q()
    
    length = [] #Used to track the length of each episode
    reward = [] #Used to track the reward obtained by each episode
    for n in range(N):
        eps = 1.0/np.sqrt(n+1) # a decreasing epsilon
        episode_reward , episode_length = play_one_game(bins,Q,eps)
        
        if n%100==0:
            print(n,'%.4f'%eps,episode_reward)
        length.append(episode_length)
        reward.append(episode_reward)
```

Q-learning is a simple and processing discrete case algorithm. It is easy to understand and also, don't need to use pytorch or tensorflow to build neural network. So it is a good Getting-Started Algorithms and Deep Reinforcement Learning is not yet involved.

Btw, the visualisation of the Q-learning is also display on the **Qlearning.ipynb** 

## Project 3: Exploring the highly encapsulated library

In these period,we have not totally understand the DRL, but while dealing with continuous cases and more complex circumstance, Q-learning and SARSA is not enough,the training effect is not that good as well.

In order to build the DRL as soon as possible , I suggest to find the library that helps us build a profound algorithms easily.

The following code is in the file **DQNtest.ipynb**

```python
import gym

from stable_baselines3 import DQN

env = gym.make("CartPole-v0")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
```

Using stable_baseline3 we can lazily start building our policy using some advanced deep reinforcement learning algorithms.

## Project4:multi-agent reinforcement learning + Stable_baseline3

First we found the official code of petting_zoo, which also happens to use stable_baseline3.The source code is in the **PistonBall.ipynb**

[Multi-Agent Deep Reinforcement Learning in 13 Lines of Code Using PettingZoo | by J K Terry | Towards Data Science](https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b)

```python
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import matplotlib.pyplot as plt

env = pistonball_v6.parallel_env(n_pistons=20,
                                 time_penalty=-0.1,
                                 continuous=True,
                                 random_drop=True,
                                 random_rotate=True,
                                 ball_mass=0.75,
                                 ball_friction=0.3,
                                 ball_elasticity=1.5,
                                 max_cycles=125
                                )

env = ss.color_reduction_v0(env, mode= 'B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

done = 0
records1 = []
env.reset()

for i in range(100):
    env.reset()
    done =0
    while not done:
        totalreward = 0
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            totalreward += reward
            act = env.action_space(agent).sample()
            env.step(act)
    records1.append(totalreward)
    print('totalreward:',totalreward,'\n')
plt.plot(records1)
plt.show()

model = PPO(CnnPolicy,
            env,
            verbose=3,
            gamma=0.95,
            n_steps=256,
            ent_coef=0.0905168,
            learning_rate=0.00062211,
            vf_coef=0.042202,
            max_grad_norm=0.9,
            gae_lambda=0.99,
            n_epochs=5,
            clip_range=0.3,
            batch_size=256
           )
model.learn(total_timesteps=2000000)
model.save('pp')



env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

model = PPO.load('pp')
done = 0
records2 = []
env.reset()
for i in range(100):
    env.reset()
    done =0
    while not done:
        totalreward = 0
        for agent in env.agent_iter():
            #env.render()
            obs, reward, done, info = env.last()
            totalreward += reward
            act = model.predict(obs, deterministic=True)[0] if not done else None
            env.step(act)
    records2.append(totalreward)
    print('totalreward:',totalreward,'\n')
plt.plot(records2)
plt.show()
```

I use this as a framework to continuously adjust the parameters and observe the changes in the training results.

After that I changed a game and changed some parameters, including the type of neural network inside.After that I successfully trained a convergent agent in another game.The code is shown in **multiwalker_v9.ipynb**

```python
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo import MultiInputPolicy
from pettingzoo.sisl import multiwalker_v9

env = multiwalker_v9.parallel_env()
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

model = PPO(
    MultiInputPolicy,
    env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.00062211,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=5,
    clip_range=0.3,
    batch_size=256,
)


model.learn(total_timesteps=2000000)
model.save("multiwalker_v9_policy_with_MIP")

# Rendering

env = multiwalker_v9.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

model = PPO.load("multiwalker_v9_policy_with_MIP")

records1 = []
for i in range(10):
    done = 0    
    env.reset()
    total = 0
    while not done:
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            act = model.predict(obs, deterministic=True)[0] if not done else None
            env.step(act)
            total += reward
    records1.append(total)
    print('episode:',i,'total reward:',total)

        
env = multiwalker_v9.env()
env = ss.frame_stack_v1(env, 3)

model = PPO.load("multiwalker_v9_policy")
records1 = []
for i in range(400):
    done = 0    
    env.reset()
    total = 0
    while not done:
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            act = model.predict(obs, deterministic=True)[0] if not done else None
            env.step(act)
            total += reward
    records1.append(total)
    print('episode:',i,'total reward:',total)

```

After that, the  classmates who is skilfully do visualization  gave me a code to visualize each reward and compare different training strategies. The code is in **show.ipynb**

```python
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.sisl import multiwalker_v9


env = multiwalker_v9.env()
env = ss.frame_stack_v1(env, 3)

model = PPO.load("multiwalker_v9_policy")
records1 = []
for i in range(400):
    done = 0    
    env.reset()
    total = 0
    while not done:
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            act = model.predict(obs, deterministic=True)[0] if not done else None
            env.step(act)
            total += reward
            env.render()
    records1.append(total)
    print('episode:',i,'total reward:',total)
    
env = multiwalker_v9.env()
env = ss.frame_stack_v1(env, 3)

model = PPO.load("multiwalker_v9_policy")
records2 = []
for i in range(400):
    done = 0    
    env.reset()
    total = 0
    while not done:
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            act = env.action_space(agent).sample()
            env.step(act)
            total += reward
    records2.append(total)
    print('episode:',i,'total reward:',total)
    
# Import pandas module and simplify it to pd
import pandas as pd
# import the pyplot module
from matplotlib import pyplot as plt

data_list = pd.DataFrame({ 'policy-1':records1,'policy-2':records2}) #Convert the list to DataFram data type, records1/2 is the data list of the two models

# Generate a canvas and set the size of the canvas to (6, 6)
plt.figure(figsize=(6, 6))

# set x/y coordinates
x = data_list.index
y = data_list.values

# draw a line chart
plt.plot(x, y,)

# Set the chart title to 'Line chart of rewards of policies' and font size to 20
plt.title('Line chart of rewards of policies',fontsize = 20)

# Set the tick font size of the axis to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set the x and y axis titles to 'expoide' and 'total reward' respectively, and the font size is 15
plt.xlabel('expoide', fontsize = 15)
plt.ylabel('total reward',fontsize = 15)

# Set the legend to 'policy-1', 'policy-2'
plt.legend(['policy-1','policy-2'])

plt.xlim((0,450)) #custom x-axis range 0, 450
plt.show()
```

## Project5: Learing pytorch and to do some regression and classification tasks.

Before completing the next step to build a deep neural network myself, I have to learn pytorch to build a neural network, at least to know the general architecture and syntax.

Here, I wrote 3 small programs, respectively, a function regression, a classification, a reinforcement learning DQN algorithm.

Regression is in file regression.py

```python
#torch is the module pytorch
import torch
#F means import excitation function
import torch.nn.functional as F
#plt Data Visualization
import matplotlib.pyplot as plt

# torch.manual_seed(1) # reproducible

#torch.squeeze() This function mainly compresses the dimension of the data
#torch.unsqueeze() This function is mainly to expand the data dimension
#generate data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)


# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module): #Inheritance
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #The constructor of the parent class is called
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output) # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x)) # activation function for hidden layer
        x = self.predict(x) # linear output
        return x

#The underlying __call__ defines: if the class name is called as a function, the forward function of the class is automatically called
net = Net(n_feature=1, n_hidden=10, n_output=1) # define the network
print(net) # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2) #SGD: stochastic gradient descent
loss_func = torch.nn.MSELoss() # this is for regression mean squared loss mean squared loss

plt.ion() # something about plotting

for t in range(200):
    #equivalent to prediction = net.predict(x)
    prediction = net(x) # input x and predict based on x
    loss = loss_func(prediction, y) # must be (1. nn output, 2. target)
    # clear gradient vector
    optimizer.zero_grad() # clear gradients for next train
    loss.backward() # backpropagation, compute gradients backpropagation
    optimizer.step() # apply gradients update parameters

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```

classification task is in file classification.py

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)#这里出来了一个二维的矩阵[100*2]，而且数值全都是1
#Returns a Tensor of random numbers drawn from separate normal distributions who’s mean and standard deviation are given.
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2) 这里的1是方差，2是均值
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2) 这里的1是方差，-2是均值
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```

DQN algorithms is in file **pytorchDQN.py**

```python

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
```

It is only a preparatory work at present, so now I have a basic understanding of the basic construction of neural networks, the relationship between deep learning and reinforcement learning.

The next task is to rebuild our multi-walker project.

## project6:Use pytorch to build Deep Reinforcement learning algorithms bottom-up.

After listening to the professor's suggestion, he said to distinguish the relationship between an agent's reward and other agents, I decided to use pytorch to build a neural network. My idea is this, the state and reward of each agent form a vector, using 4*agent number of neurons. In the neural network, each agent has corresponding neurons, but the rewards of other agents will also be back-propagated to the neurons of the current agent, which constitutes a cooperative relationship.

It's also the game multi-walker just now, but this time I use neural networks to build algorithms instead of highly encapsulated algorithm libraries.

In my vision, this neural network has 3 layers, each layer has 4*agent neurons, these neurons will process the input and output of a certain agent, in fact, it is mentioned above in the lecture: each A certain dimension of the feature vector of the agent is reflected in a certain neuron of a certain layer of neural network. But considering the identities of the three of them as collaborators, the residuals of any of the three rewards will be back-propagated to the parameters of all their neurons.

The code and its result is stored in **Project.ipynb** and **record3.txt**.

PPO.py

```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical #Create a categorical distribution with the parameter probs as the criterion, the samples are integers from "0...K-1", where K is the length of the probs parameter.
                                                                     #That is to say, according to the probability given in the incoming probs, sample at the corresponding position, and the sampling returns the integer index of the position
from torch.distributions import MultivariateNormal #multivariate-normal Multivariate normal distribution
import numpy as np

class MemoryBuffer:
    '''Simple buffer to collect experiences and clear after each update.'''
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values ​​= []
    
    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]
    
    def get_ordered_trajectories(self, n_agents=env.num_agents):#trajectories - trajectories
        ordered_actions = torch.FloatTensor()
        ordered_states = torch.FloatTensor()
        ordered_logprobs = torch.FloatTensor()
        ordered_rewards = []
        ordered_dones = []
        
        #Put one action, state, prob into the stack (but why is there no reward?)
        actions = torch.stack(self.actions)
        states = torch.stack(self.states)
        logprobs = torch.stack(self.logprobs)

        self.ordered_actions = torch.FloatTensor()
        
        for index in range(actions.shape[1]):
            if n_agents !=None and n_agents == index+1:
                break
                
            #In pytorch, there are mainly two common splicing functions, namely:
            # stack()
            # cat()
            
            ordered_states = torch.cat((ordered_states, states[:, index]), 0)
            ordered_actions = torch.cat((ordered_actions, actions[:, index]), 0)
            ordered_logprobs = torch.cat((ordered_logprobs, logprobs[:, index]), 0)
            #print("in Memory.get_order_trajectories(),\t self.rewards = ",self.rewards,'\t')
            #print("in Memory.get_order_trajectories(),\t self.dones = ",self.dones,'\t')
            ordered_rewards.extend(np.asarray(self.rewards)[:, index])
            ordered_dones.extend(np.asarray(self.dones)[:, index])

        return ordered_states, ordered_actions, ordered_logprobs, ordered_rewards, ordered_dones
        

        
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, action_std=0.5, hidden_size=32, low_policy_weights_init=True):
        super().__init__()
        
        self.actor_fc1 = nn.Linear(state_size, 4*hidden_size)
        self.actor_fc2 = nn.Linear(4*hidden_size, 4*hidden_size)
        self.actor_fc3 = nn.Linear(4*hidden_size, 2*hidden_size)
        self.actor_mu = nn.Linear(2*hidden_size, action_size)
        self.actor_sigma = nn.Linear(2*hidden_size, action_size)
        The neural network structure of #actor:
        #Input state --> 2*hidden_size neurons-->2*hidden_size neurons-->hidden_size neurons-->action_size neurons-->sigma function
        
        self.critic_fc1 = nn.Linear(state_size, 4*hidden_size)
        self.critic_fc2 = nn.Linear(4*hidden_size, 4*hidden_size)
        self.critic_fc3 = nn.Linear(4*hidden_size, 2*hidden_size)
        self.critic_value = nn.Linear(2*hidden_size, 1)
        #critic's neural network architecture
        #Input state --> 2*hidden_size neurons-->2*hidden_size neurons-->hidden_size neurons-->1 neuron
        
        self.distribution = torch.distributions.Normal
        self.action_var = torch.full((action_size,), action_std*action_std)
        
        # Boosts training performance in the beginning
        if low_policy_weights_init:
            with torch.no_grad():
                self.actor_mu.weight.mul_(0.01)

                
    def forward(self, state):
        #Use tanh as activation function
        x = torch.tanh(self.actor_fc1(state))
        x = torch.tanh(self.actor_fc2(x))
        x = torch.tanh(self.actor_fc3(x))
        mu = torch.tanh(self.actor_mu(x)) #corresponding to the mean value of the normal distribution miu
        
        #Use softplus as activation function
        sigma = F.softplus(self.actor_sigma(x))/3 #corresponding to the variance sigma of the normal distribution
        v = torch.tanh(self.critic_fc1(state))
        v = torch.tanh(self.critic_fc2(v))
        v = torch.tanh(self.critic_fc3(v))
        state_value = self.critic_value(v)
        #print("in function:forward,\tmu,sigma,state_value==",mu,sigma,state_value)
        return mu, sigma, state_value

    def act(self, state):
        '''Choose action according to the policy.'''
        #state -> nn == action_mu
        action_mu, action_sigma, state_value = self.forward(state)
        # action_mu
        action_var = self.action_var.expand_as(action_mu)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        #print("in function:act,\taction.detach(), log_prob.detach()==",action.detach(), log_prob.detach())
        return action.detach(), log_prob.detach()
    def evaluateStd(self, state, action):
        '''Evaluate action using learned std value for distribution.'''
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu.squeeze(), action_sigma.squeeze())
        log_prob = m.log_prob(action)

        return log_prob, state_value

    def evaluate(self, state, action):
        '''Evaluate action for a given state.'''   
        action_mean, action_var, state_value = self.forward(state)
        #print("in function evaluate:action_mean,action_var=",action_mean,action_var)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO():
    '''Proximal Policy Optimization algorithm.'''
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_clip=0.2, epochs=20, action_std=0.5):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma  = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = epochs

        self.policy = ActorCritic(self.state_size, self.action_size, action_std, hidden_size=128)
        self.policy_old = ActorCritic(self.state_size, self.action_size, action_std, hidden_size=128)

        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.episode = 0
    
    def select_action(self, state):
        '''Get action using state in numpy format'''
        # state = torch.FloatTensor(state.reshape(1, -1))
        state = torch.FloatTensor(state)
        #print("in function:select_action,\tself.policy_old.act(state)==",self.policy_old.act(state))
        return self.policy_old.act(state)

    def update(self, memory):
        '''Update agent's network using collected set of experiences.'''
        states, actions, log_probs, rewards, dones = memory.get_ordered_trajectories(env.num_agents)

        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i] == True:
                discounted_reward = 0  
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # old_state_values = torch.stack(state_values, 1).detach()
        # advantages = discounted_rewards - old_state_values.detach().squeeze()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        # states = torch.squeeze(torch.stack(states), 1).detach()
        # actions = torch.squeeze(torch.stack(actions), 1).detach()
        # old_log_probs = torch.squeeze(torch.stack(log_probs), 1).detach()

        states = states.detach()
        actions = actions.detach()
        old_log_probs = log_probs.detach()


        for epoch in range(self.K_epochs):

            new_log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            new_log_probs = new_log_probs.squeeze()
            advantages = discounted_rewards - state_values.detach().squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon_clip, max=1+self.epsilon_clip)
            loss = -torch.min(ratios*advantages, ratios_clipped*advantages)+ 0.5*self.MseLoss(state_values, discounted_rewards) - 0.01*dist_entropy
            loss = torch.tensor(loss,dtype=float,requires_grad=True)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
```

train.py

```python
import torch                           
import numpy as np          
from collections import deque                                   #队列的数据结构
import time                     
from torch.utils.tensorboard import SummaryWriter               #这个包的作用是让使用者可以通过简单的界面用PyTorch获取events，
from pettingzoo.sisl import multiwalker_v9                      #并且在tensorboard中显示。


n_agents = 3
n_episodes = 12000
max_steps = 300
update_interval = 1200/n_agents
log_interval = 10
solving_threshold = 30
time_step = 0
render = True
train = True
pretrained = False
tensorboard_logging = True

env_name = "multiwalker_v9_new3"
env = multiwalker_v9.parallel_env(forward_reward=50,position_noise=0, angle_noise=0,fall_reward=-10)

obsagents= env.reset()
action_size = 4
states = obs
state_size = 31

print("State size: ", state_size)
print("Action size: ", action_size)

scores = deque(maxlen=log_interval)
max_score = -1000
episode_lengths = deque(maxlen=log_interval)
rewards =  []

memory = MemoryBuffer()

agent = PPO(state_size, action_size)

if not train:
    agent.policy_old.eval()
else:
    print('logs/'+env_name+'_'+str(time.time()))

if pretrained:
    agent.policy_old.load_state_dict(torch.load('./'+env_name+'_model_best_old.pth'))
    agent.policy.load_state_dict(torch.load('./'+env_name+'_model_best_old.pth'))
doc = open('record3.txt',mode='r+',buffering=1)
for n_episode in range(1, n_episodes+1):
    obs = env.reset()
    states = list(obs.values())
    #state = states[0]
    
    states = torch.FloatTensor(states)
    # print("States shape: ", states.shape)
    # state = torch.FloatTensor(state.reshape(1, -1))
    
    episode_length = 0
    episodic_rewards = []
    done = 0
    total_rewards = []
    for t in range(max_steps):
        
        time_step += 1

        actions, log_probs = agent.select_action(states)
        #print("actions ==",actions,file=doc)

        states = torch.FloatTensor(states)
        memory.states.append(states)
        memory.actions.append(actions)
        memory.logprobs.append(log_probs)

        # actions = []
        # ## Unity env style
        # for agent_id in range(0,20):
        #     actions.append(action.data.numpy().flatten())

        obs, rewards, dones, infos = env.step(dict(zip(env.agents,actions.data.numpy().clip(-1,1))))          # send all actions to tne environment
        states = list(dict(obs).values())
        rewards = list(dict(rewards).values())
        dones = list(dict(dones).values()) 
        #print("obs, rewards, dones, infos = \n",obs,'\n', rewards,'\n', dones,'\n', infos,file=doc)
        

        
        
        #state = states[0]
        #reward = rewards[0]
        done = dones[0]

        # state, reward, done, _ = env.step(action.data.numpy().flatten())


        memory.rewards.append(rewards)
        #print('in class:memory,memory.rewards=',memory.rewards)
        memory.dones.append(dones)
        episodic_rewards.append(rewards)
        state_value = 0
        
        if render:
            env.render()
             #image = env.render(mode = 'rgb_array')
             #if time_step % 2 == 0:
             #writerImage.append_data(image)

        if train:
            if time_step % update_interval == 0:
                
                agent.update(memory)
                time_step = 0
                memory.clear_buffer()

        episode_length = t

        if done:
            break
            
    
    episode_lengths.append(episode_length)
    total_reward = np.sum(episodic_rewards)/n_agents
    print('total_rewards =',total_reward,file=doc)
    scores.append(total_reward)
    
    if train:
        if n_episode % log_interval == 0:
            print("Episode: ", n_episode, "\t Avg. episode length: ", np.mean(episode_lengths), "\t Avg. score: ", np.mean(scores))

            if np.mean(scores) > solving_threshold:
                print("Environment solved, saving model")
                torch.save(agent.policy_old.state_dict(), 'PPO_model_solved_{}.pth'.format(env_name))
        
        if n_episode % 100 == 0:
            print("Saving model after ", n_episode, " episodes")
            torch.save(agent.policy_old.state_dict(), '{}_model_{}_episodes.pth'.format(env_name, n_episode))
            
        if total_reward > max_score:
            print("Episode: ", n_episode,'\t')
            print("Saving improved model")
            max_score = total_reward
            torch.save(agent.policy_old.state_dict(), '{}_model_best.pth'.format(env_name))

        if tensorboard_logging:
            print('Score', {'Score':total_reward, 'Avg._Score': np.mean(scores)}, n_episode)
            print('Episode_length', {'Episode_length':episode_length, 'Avg._Episode length': np.mean(episode_lengths)}, n_episode)
            #writer.add_scalars('Score', {'Score':total_reward, 'Avg._Score': np.mean(scores)}, n_episode)
            #writer.add_scalars('Episode_length', {'Episode_length':episode_length, 'Avg._Episode length': np.mean(episode_lengths)}, n_episode)
    
    else:
        print("Episode: ", n_episode, "\t Episode length: ", episode_length, "\t Score: ", total_reward)
        
    total_reward = 0
print(scores)
```

However, the final training effect of this project is not ideal, and the training of the agent does not actually improve. At first I thought it was a reward issue, so I increased the penalty for dropped packages, hoping to make the robot last longer. However, the result was frustrating, the villain still did not improve much. Might be the cause of something wrong with my design, but I haven't thought of a relevant solution so far.

An improved approach might be to separate the input of each agent, and then let the neurons belonging to the agent process the input and output of the agent independently. However, this disadvantage is that it is difficult to achieve Nash equilibrium by cutting off their cooperative relationship.

It may be that I have not grasped the essence of multi-agent in place, and only use traditional PPO to try to solve a game environment where three agents interact with each other.

# WRITING IN THE LAST:

I am very sorry to have failed the last most important task. The above documents record my journey from never hearing about deep learning and reinforcement learning to getting started with RL, DL and DRL. Of course, this can also be used as a reference for others to get started in the future.

The above is the realization of the knowledge I learned in this summer project with code. I am very grateful for the guidance of the professor. I hope that I can still write an email to you for advice on AI-related projects in the future.  
  
