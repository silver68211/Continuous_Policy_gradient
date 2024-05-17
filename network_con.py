import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.layers as layers 

class ConPolicyGrad(keras.Model):
    def __init__(self, fc1_dims = 5, fc2_dims = 5, 
                 fc3_dims = 5, out1_dims=1,
                 out2_dims=1, learn_var= False, fix_var = 0.1):
        super(ConPolicyGrad, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.out1_dims = out1_dims
        self.out2_dims = out2_dims
        self.learn_var = learn_var

        self.fc1 = layers.Dense(self.fc1_dims, activation= 'relu', kernel_initializer='glorot_uniform')
        self.fc2 = layers.Dense(self.fc2_dims, activation = 'relu', kernel_initializer='glorot_uniform')
        self.mn  = layers.Dense(self.out1_dims, activation = 'linear', kernel_initializer='glorot_uniform')

        if learn_var:
        #    self.fc3 = layers.Dense(self.fc3_dims, activation = 'relu', kernel_initializer='glorot_uniform')
            self.var = layers.Dense(self.out2_dims, activation = 'softplus', 
                                    kernel_initializer='glorot_uniform')
    def call(self, state):
        
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mn(x)
        if self.learn_var:
            # x = self.fc3(x)
            var = self.var(x) + 0.5 
            
            return mu, var
        else:
            return mu
        
