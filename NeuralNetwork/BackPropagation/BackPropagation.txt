A two layer perceptron with the backpropagation algorithm has been implemented to solve the parity problem. 
The desired output for the parity problem is 1 if an input pattern contains an odd number of 1's and 0 otherwise. 
The learning procedure is stopped when an absolute error (difference) of 0.05 is reached for every input pattern.

Input Layer has 4 input points, Hidden Layer has 4 neurons and the Output Layer has 1 neuron.

Input Data:
[0 0 0 0]
[0 0 0 1]
[0 0 1 0]
[0 0 1 1]
[0 1 0 0]
[0 1 0 1]
[0 1 1 0]
[0 1 1 1]
[1 0 0 0]
[1 0 0 1]
[1 0 1 0]
[1 0 1 1]
[1 1 0 0]
[1 1 0 1]
[1 1 1 0]
[1 1 1 1]

Hidden Layer Bias
[ 0.08680988 -0.44326123 -0.15096482  0.68955226]

Hidden Layer Weights
[-0.99056229 -0.75686176  0.34149817  0.65170551]
[-0.72658682  0.15018666  0.78264391 -0.58159576]
[-0.62934356 -0.78324622 -0.56060501  0.95724757]
[ 0.6233663  -0.65611797  0.6324495  -0.45185251]

Output Layer Bias
[0] [0] [0] [0]

Output Layer Weights
[-0.13659163]
[ 0.88005964]
[ 0.63529876]
[-0.3277761 ]

Expected Output
[0 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0]
