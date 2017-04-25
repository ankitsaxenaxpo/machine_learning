# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:13:12 2017

@author: Ankit
"""

import itertools
import numpy as np

#initialize input size
inputs_n1 = 4
hidden_n1 = 4
output_n1 = 1
learning_rate = 0.05

#initialize input array
inputs = []

for temp in list(itertools.combinations([1,0,1,0,1,0,1,0], 4)):
    a = []
    a.append(1)
    for e in temp:
        a.append(e)
    if a not in inputs:
        inputs.append(a)
inputs = np.array(sorted(inputs))

#expected output
outputs_expected = [1 if sum(inputs[i])%2 == 0 else 0 for i in range(len(inputs))]
outputs_expected = np.array(outputs_expected)
                    
outputs = {}

class NN:
    
    #initialize parameters
    eta = 0.05
    alpha = 0.5
    threshold_error = 0.05
    
    def __init__(self, i, h, hw, o, ow, eta):
        print "\nInside Constructor"
        #assign random weights
        self.eta = eta
        self.inputs_n = i
        
        self.hidden_n = h
        self.hidden_weights = hw
        
        self.output_n = o
        self.output_weights = ow
        
    #activation function
    def sigmoid_calc(self, sum):
        return 1/(1 + np.exp(-sum))
        
    #derivative of activation function    
    def sigmoid_derivative(self, sum):
        return np.exp(-sum)/((1 + np.exp(-sum))**2)
        
    #forward propagation    
    def forward_unit(self, input_n):
        #print "------Inside Forward Propagation Method"
        
        #hidden_input is the input to th hidden layer
        self.hidden_input = np.dot(input_n, self.hidden_weights)
        
        #hidden_output is the output of hidden layer
        self.hidden_output = self.sigmoid_calc(self.hidden_input)
        
        #output_input is the input to the output layer
        self.output_input = np.dot(self.hidden_output, self.output_weights)
        #print "Output is : ", self.output_input
        
        #predicted is the output of the output layer
        predicted = self.sigmoid_calc(self.output_input)
        return predicted
        
    def error_cal(self, inputs1, expected):
        #print "----Inside Error Calculation Method"
        
        self.actual = self.forward_unit(inputs1)
        return np.abs((expected - self.actual[:,0]))
        
    def backward_unit(self, expected, actual):
        #print "--------Inside Backward Propagation Method"
        
        #output layer: backward propagation
        self.output_delta = np.multiply(-(expected - actual[:,0]), self.sigmoid_derivative(self.output_input)[:,0])
        self.output_delta.shape = (16,1)
        
        #hidden layer: backward propagation
        self.hidden_delta = np.multiply(np.dot(self.output_delta, self.output_weights.T), self.sigmoid_derivative(self.hidden_input))
        
    def update_weights(self, input_n):
        #print "----------Inside Update Weights Method"
        
        self.hidden_weights -= (self.eta / (1 - self.alpha)) * np.dot(input_n.T, self.hidden_delta)
        self.output_weights -= (self.eta / (1 - self.alpha)) * np.dot(self.hidden_output.T, self.output_delta)
        
    def forward_backward(self, inputs, expected):
        print "\n--Inside Foward-Backward Method"
        for epoch in range(0, 1000000):
            error = self.error_cal(inputs, expected)            
            if(sum(error < self.threshold_error) == len(inputs)):
                print "Final Hidden Bias and Weights: \n", self.hidden_weights
                print "Final Output Weights: \n", self.output_weights
                print "Actual Output: \n", self.actual
                print "epoch: ", str(epoch+1), "eta: ", self.eta, ", error: ", error
                break
                
            else:
                #print "epoch: ", str(epoch+1), "eta: ", self.eta
                self.backward_unit(expected, self.actual)
                self.update_weights(inputs)
                       
if __name__ == "__main__":
    
    print "\nInput: \n", inputs
    print "\nExpected Output: \n", outputs_expected
    
    while learning_rate <= 0.5:
        print "******************learning rate: ", learning_rate, " ******************"
        np.random.seed(100)
        hw = np.random.uniform(-1, 1, (inputs_n1 + 1, hidden_n1))
        ow = np.random.uniform(-1, 1 ,(hidden_n1, output_n1))
        
        print "\nHidden Bias and Weights: \n", hw
        print "\nOutput Weights: \n", ow
        
        bp = NN(inputs_n1, hidden_n1, hw, output_n1, ow, learning_rate)
        bp.forward_backward(inputs, outputs_expected)
        learning_rate += 0.05
        