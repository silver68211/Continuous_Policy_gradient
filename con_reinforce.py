import tensorflow as tf
import tensorflow_probability as tfp 
from tensorflow.keras.optimizers import Adam

import numpy as np

from network_con import ConPolicyGrad

class Agent: 
    def __init__(self, fc1_dims = 256, fc2_dims = 256, 
                 fc3_dims = 256, out1_dims = 1, out2_dims= 1,
                 fix_var = 0.1, n_sample = 1, 
                 alpha = 0.001, gamma = 1, 
                 learn_var = False):
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.gamma = gamma

        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []

        self.out1_dims = out1_dims 
        self.out2_dims = out2_dims

        self.learn_var = learn_var
        self.fix_var = fix_var
        self.n_sample = n_sample
        self.alpha = alpha
        self.fix_var = 0.1

        self.policy = ConPolicyGrad( fc1_dims= self.fc1_dims, fc2_dims= self.fc2_dims, 
                                    fc3_dims= self.fc3_dims, out1_dims= self.out1_dims,
                                    out2_dims= self.out2_dims, learn_var= self.learn_var,
                                    fix_var= self.fix_var)
        
        self.policy.compile(optimizer=Adam(learning_rate = self.alpha))
    
    def step(self, state):
        if self.learn_var:
            mu, std = self.policy(state)
            return mu, std
        else:
            mu = self.policy(state)
            return mu

    def choose_action(self, mu, std = 0.1):
        
        action = tf.random.normal([self.n_sample], mean = mu, stddev=std)
        return action

        
        

    def reward(self, action):
        mu_target = tf.convert_to_tensor([4], dtype=tf.float32)
        target_range = tf.convert_to_tensor([0.5], dtype=tf.float32)
        max_reward = tf.convert_to_tensor([1], dtype=tf.float32)
        
        r = max_reward/max(target_range, abs(mu_target-action))*target_range
        
        return r
    
    def lossFun(self, state, action, reward):

        if self.learn_var: 
            mu, std = self.policy(state)
            # print('mu: ', mu)
            # print('action: ', action)
            # print('std: ', std)
            
            p = tf.exp(-(action-mu)**2/(2*std**2))/(std*tf.sqrt(2*np.pi))
            # print('p: ', p)
            log_p = tf.math.log(p+1e-5)
            # print('reward: ', reward)
            # print('reward: ', log_p)
            # print('p', std)
            # print('loss: ', -reward*log_p)
            return -reward*log_p
        else: 
            std = self.fix_var
            mu = self.policy(state)
            p = tf.exp(-0.5*((action-mu)/(std))**2)/(std*tf.sqrt(2*np.pi))
            log_p = tf.math.log(p+1e-5)
            return -reward*log_p
        
    def store_transitions(self, state, action, reward):
        self.state_memory.append(state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)

    


    def learn(self):

        

        G = np.zeros_like(self.reward_memory)

        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        
        with tf.GradientTape() as tape:
            loss = 0
            
            for state, action, reward in zip(self.state_memory,self.action_memory, self.reward_memory):
                
                
                loss += self.lossFun(state, action, reward)
            
            # print('loss: ', loss)

            grads = tape.gradient(loss, self.policy.trainable_variables)
            
            self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))


        self.state_memory  = []
        self.reward_memory = []
        self.action_memory = []
            


        



        