#coding=gbk
import gym                               #openAI gym�Ŀ�
import numpy as np                       #����numpy����������;��������
from gym import wrappers                  # �������԰����ǰ���Ϸ����Ƶ�����Ƶ��ʽ

env= gym.make('CartPole-v0')              #ѡ��ѵ������Ϸ

bestlength = 0                            #���������õ����->�������Ĵ���
episode_length = []                       #�������ÿһ�εĳ����Ĳ���

best_weight = np.zeros(4)                 #4ά���飺��������������£��������������ã�Ȩ�أ���

for i in range(100):                    #����100��ѵ�� ÿ��ѵ����100����Ϸ����һ��forѭ����
    
    new_weight = np.random.uniform(-1.0,1.0,4) #ÿһ�ζ��Ѳ�����Ȩ�أ������վ��ȷֲ���ʼ��
    
    length = []                           #length������¼��һ�εĽ��
    
    for j in range(100):
        #��ʼ����������
        
        observation = env.reset()
        done = False
        cnt = 0
        
        while not done:  #һ����Ϸ
            #env.render()                      #�����ע�͵�����Ϊ10000��render̫�˷�ʱ��
        
            cnt += 1
            #����������˼�ǣ��������������Ȩ�أ��뻷�����������ĵ�˻�����0��ô��ȡaction[1] �����ȡaction[2]
            action =1 if np.dot(observation,new_weight) >0 else 0
            #��������������actionͨ��step���������ȥ
            observation,reward,done,_ = env.step(action)
            
            if done:                 #�����Ϸ�����ˣ���ô���˳����100����ѭ��
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
    
        
print('with the best weight, game last %d'%cnt,' move')                        #���agent�����˶��ٴ�
        
