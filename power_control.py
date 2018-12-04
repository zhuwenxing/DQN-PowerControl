import random
import numpy as np
import math

class GameState:
    def __init__(self,P_1,P_2,Noise,Sensor):
        
        self.P_1 = P_1
        self.P_2 = P_2
        self.length_P_1 = len(self.P_1)
        self.length_P_2 = len(self.P_2)
        
        self.h_11 = 1;self.h_12 = 1
        self.h_21 = 1;self.h_22 = 1
        
        self.ita_1 = 1.2
        self.ita_2 = 0.7
        
        self.sigma_sq_1 = 0.01
        self.sigma_sq_2 = 0.01
        
        self.lam = 0.1
        self.alpha = 0.5

        self.num_sensor = Sensor
        self.noise = Noise
        
        self.P,self.sigma = self.dis()
        
    def dis(self):
        d = [[random.uniform(100,300) for _ in range(self.num_sensor)] for _ in range(2)]
        P = np.zeros((2,self.num_sensor))
        for i in range(0,self.num_sensor):
            P[0][i] = ((self.lam/(4*math.pi)/d[0][i])**2)
            P[1][i] = ((self.lam/(4*math.pi)/d[1][i])**2)
        sigma = np.zeros((self.num_sensor))
        for i in range(0,self.num_sensor):
            sigma[i] = ( P[0][i]*self.P_1[0]+P[1][i]*self.P_2[0] )/ self.noise
        return P,sigma
    
    def ini(self):
        self.p_1 = self.P_1[random.randint(0,self.length_P_1-1)]
        self.p_2 = self.P_2[random.randint(0,self.length_P_2-1)]

    def ini_test(self):
        self.p_1_test = self.P_1[random.randint(0,self.length_P_1-1)]
        self.p_2_test = self.P_2[random.randint(0,self.length_P_2-1)]
    
    def frame_step(self, input_actions, policy, i):
        if i == True:
            if policy == 1:
                self.p_1 = self.update_p1_v1(self.p_2)
            if policy == 2:
                self.p_1 = self.update_p1_v2(self.p_1,self.p_2)
            action = np.flatnonzero(input_actions)[0]   # Return indices that are non-zero in the flattened version of a.
            # 返回非零元素的下标。
            self.p_2 = self.P_2[action]
        observation = self.compute_observation(self.p_1,self.p_2)
        reward = self.compute_reward(self.p_1,self.p_2)
        
        terminal = (reward==10)
        
        return observation,reward,terminal
    
    def frame_step_test(self, input_actions, policy, i):
        if i == True:
            if policy == 1:
                self.p_1_test = self.update_p1_v1(self.p_2_test)
            if policy == 2:
                self.p_1_test = self.update_p1_v2(self.p_1_test,self.p_2_test)
            action = np.flatnonzero(input_actions)[0]
            self.p_2_test = self.P_2[action]
        observation = self.compute_observation(self.p_1_test,self.p_2_test)
        reward = self.compute_reward(self.p_1_test,self.p_2_test)
        
        terminal = (reward==10)  # 当reward==10时，作为terminal的标志。
        
        return observation,reward,terminal
    
    def compute_observation(self,x,y):   # 
        observation = np.zeros((self.num_sensor))
        for i in range(0,self.num_sensor):
            observation[i] = self.P[0][i] * x + self.P[1][i] * y + random.gauss(0,self.sigma[i])
            if observation[i]<0:
                observation[i] =0
            observation[i] = observation[i]*(10**7)
        return observation
    
    def compute_reward(self,x,y):
        success_1,success_2 = self.compute_SINR(x,y)
        reward = self.alpha * success_1 + (1-self.alpha) * success_2
        if reward == 0.5:
            reward = 0
        if reward == 1:
            reward = 10
        return reward
    
    def update_p1_v1(self,y):
        p_1_n = self.ita_1/((abs(self.h_11)**2)/((abs(self.h_21)**2)*y + self.sigma_sq_1))
        v = []
        for ind in range(self.length_P_1):
            v.append(max(p_1_n-self.P_1[ind],0))
        p_1_new = self.P_1[v.index(min(v))]
        return p_1_new
    
    def update_p1_v2(self,x,y):
        ind_p_1 = self.P_1.index(x)
        tSINR_1 = ((abs(self.h_11)**2)*x/((abs(self.h_21)**2)*y + self.sigma_sq_1))
        tao = x * self.ita_1 / tSINR_1
        if tao>=x and ind_p_1+1<=self.length_P_1-1 and tao<=self.P_1[ind_p_1+1] :
            x = self.P_1[ind_p_1+1]
        elif ind_p_1-1>=0 and tao<=self.P_1[ind_p_1-1] :
            x = self.P_1[ind_p_1-1]
        return x
    
    def compute_SINR(self,x,y):
        success_1 = ( (abs(self.h_11)**2)*x/((abs(self.h_21)**2)*y + self.sigma_sq_1)) >= self.ita_1
        success_2 = ( (abs(self.h_22)**2)*y/((abs(self.h_12)**2)*x + self.sigma_sq_2)) >= self.ita_2
        return success_1,success_2
