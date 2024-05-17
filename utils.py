import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, filename, x=None, window=5, xlabel= 'Game', ylabel='Score'): 
    plt.figure()  
    N = len(scores)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
         x = [i for i in range(N)]

    plt.ylabel(ylabel)       
    plt.xlabel(xlabel)                     
    plt.plot(x, running_avg)
    plt.grid()
    plt.savefig(filename)
    