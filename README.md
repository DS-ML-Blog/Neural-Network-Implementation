# Artificial Neural Network implementation

This project is an implementation of a feedforward Artificial Neural Network without using high-level deep learning frameworks like Tensorflow of PyTorch.  There is one hidden layer in the network. Number of neurons in input, hidden and output layer is adjustable.  For finding optimal hyperparameters (number of hidden neurons and size of training data) a simple grid search algorithm was employed. 



### Training data

As the input and output data any sequences of numbers may be used. Feedforward Artificial Neural Network can handle them, no matter what is the interpretation. Originally summing and averaging four numbers was used for simplicity.



### Algorithm description

Optimization algorithm for training the network is backpropagation. Pipeline used in this project is the following (for each combination of hidden neurons number and training set size):

- Initialize weights and biases 
- Initialize matrices of each neuron outputs and pre-outputs (values before applying activation function)
- Generate training set
- Calculate output values of all network's neurons
- Calculate vector of cost function gradient
- Calculate new set of weights and biases
- Continue until convergence
- Calculate outputs for a test set and evaluation metrics (R2, MSE, MAE, MAPE)