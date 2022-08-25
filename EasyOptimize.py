#coding=gbk
import gym                               #openAI gym的库
import numpy as np                       #利用numpy库进行向量和矩阵的运算
from gym import wrappers                  # 这个库可以帮我们把游戏的视频变成音频格式

env= gym.make('CartPole-v0')              #选择训练的游戏

bestlength = 0                            #储存的是最好的情况->持续最多的次数
episode_length = []                       #储存的是每一次的持续的步数

best_weight = np.zeros(4)                 #4维数组：储存的是最好情况下，各个参数的配置（权重？）

for i in range(100):                    #进行100个训练 每次训练玩100次游戏（下一个for循环）
    
    new_weight = np.random.uniform(-1.0,1.0,4) #每一次都把参数（权重？）按照均匀分布初始化
    
    length = []                           #length用来记录这一次的结果
    
    for j in range(100):
        #初始化环境参数
        
        observation = env.reset()
        done = False
        cnt = 0
        
        while not done:  #一次游戏
            #env.render()                      #把这个注释掉是因为10000次render太浪费时间
        
            cnt += 1
            #下面这句的意思是：如果参数向量（权重）与环境参数向量的点乘积大于0那么采取action[1] 否则采取action[2]
            action =1 if np.dot(observation,new_weight) >0 else 0
            #将上面计算得来的action通过step函数输入进去
            observation,reward,done,_ = env.step(action)
            
            if done:                 #如果游戏结束了，那么就退出这个100步的循环
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
cnt =0
observation = env.reset()

while not done:
    cnt +=1
    action = 1 if np.dot(observation,best_weight)>0 else 0
    observation,reward,done,_ = env.step(action)
    
    if done:
        break
    
        
print('with the best weight, game last %d'%cnt,' move')                        #输出agent操作了多少次
        
