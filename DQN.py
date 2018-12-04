import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

GAMMA = 0.8  
OBSERVE = 300 
EXPLORE = 100000 
FINAL_EPSILON = 0.0 
INITIAL_EPSILON = 0.8 
REPLAY_MEMORY = 400 
BATCH_SIZE = 256 

class BrainDQN:

	def __init__(self,actions,Sensor):
        
		self.replayMemory = deque()
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.recording = EXPLORE
		self.sensor_dim = Sensor
		self.actions = actions
		self.hidden1 = 256
		self.hidden2 = 256
		self.hidden3 = 512
        
		self.createQNetwork()

	def createQNetwork(self):

		W_fc1 = self.weight_variable([self.sensor_dim,self.hidden1])
		b_fc1 = self.bias_variable([self.hidden1])

		W_fc2 = self.weight_variable([self.hidden1,self.hidden2])
		b_fc2 = self.bias_variable([self.hidden2])
        
		W_fc3 = self.weight_variable([self.hidden2,self.hidden3])
		b_fc3 = self.bias_variable([self.hidden3])
        
		W_fc4 = self.weight_variable([self.hidden3,self.actions])
		b_fc4 = self.bias_variable([self.actions])        

		self.stateInput = tf.placeholder("float",[None,self.sensor_dim])

		h_fc1 = tf.nn.relu(tf.matmul(self.stateInput,W_fc1) + b_fc1)
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)
		h_fc3 = tf.nn.tanh(tf.matmul(h_fc2,W_fc3) + b_fc3)        
        
		self.QValue = tf.matmul(h_fc3,W_fc4) + b_fc4

		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
		self.trainStep = tf.train.AdamOptimizer(learning_rate=10**-5).minimize(self.cost)

		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())

	def trainQNetwork(self):
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		y_batch = []
		QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})
		for i in range(0,BATCH_SIZE):

			y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		_, self.loss = self.session.run([self.trainStep,self.cost],feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})
		return self.loss

	def setPerception(self,nextObservation,action,reward):
		loss = 0
		newState = nextObservation
		self.replayMemory.append((self.currentState,action,reward,newState))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
            
			loss = self.trainQNetwork()

		self.currentState = newState
		self.timeStep += 1
		return loss
        

	def getAction(self):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})
		action = np.zeros(self.actions)
		if random.random() <= self.epsilon:
			action_index = random.randrange(self.actions)
			action[action_index] = 1
		else:
			action_index = np.argmax(QValue)
			action[action_index] = 1
         
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
			self.recording = self.recording-1

		return action,self.recording
    
	def getAction_test(self,observation):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[observation]})
		action = np.zeros(self.actions)
		action_index = np.argmax(QValue)
		action[action_index] = 1

		return action
    
	def setInitState(self,observation):
		self.currentState = observation

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)
            