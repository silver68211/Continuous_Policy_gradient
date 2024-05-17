import tensorflow as tf
import numpy as np
from con_reinforce import Agent
from utils import plotLearning

learn_var = True
std = 0.1

if __name__ == '__main__':
    agent = Agent(fix_var=0.1, alpha=0.003, gamma= 1, 
                  learn_var=learn_var)

    score_history = []
    mu_history = []
    sigma_history =[]

    num_episiod = 10000+1
    
    for i in range(num_episiod):
        score = 0
        state = tf.convert_to_tensor([[1]], dtype=tf.float32)
        done = 1

        while done:
            if learn_var:
                mu, std = agent.step(state)
                
            else: 
                mu = agent.step(state)
            
            
            action = agent.choose_action(mu=mu, std=std)
            
            reward = agent.reward(action=action)
            agent.store_transitions(state, action, reward)
            state = mu
            score += reward
            done -= 1
        
        
        score_history.append(score)
        mu_history.append(mu)
        sigma_history.append(std)
        

        agent.learn()


        avg_score = np.mean(score_history[-100:])
        print('episode: ', i,'score: %.1f' % score,
            'average score %.1f' % avg_score)

    filename = 'score.png'
    plotLearning(score_history, filename=filename, window=100)
    plotLearning(mu_history, filename='mu.png', window=100, ylabel='Mean')
    plotLearning(sigma_history, filename='sigma.png', window=100, ylabel='Sigma')


        



