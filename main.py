from power_control import GameState
from DQN import BrainDQN
import numpy as np
import matplotlib.pyplot as plt

P_1 = [round(0.1*i/2.0,2) for i in range(1,9)]
P_2 = [round(0.1*i/2.0,2) for i in range(1,9)]
actions = len(P_2)

Loss = []
Success = []
Fre = []

noise = 3      
num_sensor = 10  # N 
policy = 2      # choose power change policy for PU, it should be 1(Multi-step) or 2(Single step)

brain = BrainDQN(actions,num_sensor)
com = GameState(P_1,P_2,noise,num_sensor)
terminal =True
recording = 100000

while(recording>0):
    # initialization
    if(terminal ==True):
        com.ini()
        observation0, reward0, terminal = com.frame_step(np.zeros(actions),policy,False)
        brain.setInitState(observation0)

    # train
    action,recording = brain.getAction()
    nextObservation,reward,terminal = com.frame_step(action,policy,True)
    loss = brain.setPerception(nextObservation,action,reward)

    # test
    if (recording+1)%500==0:
        
        Loss.append(loss)
        print "iteration : %d , loss : %f ." %(100000-recording, loss)
        
        success = 0.0
        fre = 0
        num = 1000.0
        for ind in range(1000):
            T = 0
            com.ini_test()
            observation0_test, reward_test, terminal_test = com.frame_step_test(np.zeros(actions),policy,False)
            while (terminal_test !=True) and T<20:
                action_test = brain.getAction_test(observation0_test)
                observation0_test,reward_test,terminal_test = com.frame_step_test(action_test,policy,True)
                T +=1
            if terminal_test==True:
                success +=1
                fre +=T
        if success == 0:
            fre = 0
        else:
            fre = fre/success
        success = success/num
        Success.append(success)
        Fre.append(fre)
        print "success : %f , step : %f ." %(success , fre)
        
plt.plot(Loss)
plt.show()

plt.plot(Success)
plt.show()

plt.plot(Fre)
plt.show()
