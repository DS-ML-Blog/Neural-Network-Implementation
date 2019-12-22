import numpy as np


def activationFunction(x, loc_vector):

    x = list(x)

    layers_vs_activation_functions = {0: 'None', 1: 'ReLu', 2: 'None'}

    if loc_vector[0] == 0:

        if layers_vs_activation_functions[loc_vector[1]] == 'None':
            y = x
        elif layers_vs_activation_functions[loc_vector[1]] == 'ReLu':
            y = [max(i, 0) for i in x]
        elif layers_vs_activation_functions[loc_vector[1]] == 'Sigm':
            y = np.exp(x)/(np.exp(x)+1)

    else:

        if layers_vs_activation_functions[loc_vector[1]] == 'None':
            y = [1]*len(x)
        elif layers_vs_activation_functions[loc_vector[1]] == 'ReLu':
            y = np.heaviside(x, 0.5)
        elif layers_vs_activation_functions[loc_vector[1]] == 'Sigm':
            y = np.exp(x)/(np.exp(x)+1)**2

    return np.array(y)


def generateTrainingSet(n_training, n_inputs, input_limits):

    dataMatrix = np.zeros([n_training, n_inputs+2])

    for i in range(n_training):

        inputs = np.zeros(n_inputs)
        for j, lims in enumerate(input_limits):
            inputs[j] = np.random.randint(lims[0], lims[1])

        dataMatrix[i, 0:n_inputs] = inputs
        dataMatrix[i, n_inputs] = np.sum(inputs)
        dataMatrix[i, n_inputs+1] = np.sum(inputs)/n_inputs

    return dataMatrix


def getNetworkOutput(y1, W, n_inputs, n_hidden, n_outputs):
    W1 = W[0:n_hidden, 0:n_inputs+1, 0]
    W2 = W[0:n_outputs, 0:n_hidden+1, 1]

    z1 = activationFunction(y1, [0, 0])

    y2 = np.matmul(z1.transpose(), W1[:, :-1].transpose()) + W1[:, -1]
    z2 = activationFunction(y2, [0, 1])

    y3 = np.matmul(z2.transpose(), W2[:, :-1].transpose()) + W2[:, -1]
    z3 = activationFunction(y3, [0, 2])

    return z3


def calcCost(trainingSet, W, n_inputs, n_hidden, n_outputs):
    W1 = W[0:n_hidden, 0:n_inputs+1, 0]
    W2 = W[0:n_outputs, 0:n_hidden+1, 1]

    costs = []     
    for i in range(np.shape(trainingSet)[0]): 
        unit_cost = 0       

        for j in range(n_inputs, np.shape(trainingSet)[1]):
            unit_cost += (trainingSet[i, j]
                - getNetworkOutput(trainingSet[i, 0:n_inputs], W, n_inputs,
                                   n_hidden, n_outputs)[j-n_inputs])**2

        costs.append(unit_cost)

    return np.mean(costs)


def calcGradElement(n, Z, W, y_n, Y, case, bound, n_inputs):
    chain_rule1 = 2*(Z[n, 2] - y_n)
    chain_rule2 = activationFunction([Y[n, 2]], [1, 2])

    if case == [0, 0]:
        chain_rule3 = 0
    elif case == [1, 0]:
        chain_rule3 = 0
    elif case == [0, 1]:
        m = bound[1]
        chain_rule3 = Z[m, 1]
    elif case == [1, 1]:
        chain_rule3 = 1
    elif case == [0, 2]:
        suma = np.dot(Z[0:n_inputs, 0], W[bound[0], 0:n_inputs, 0]) \
               + W[bound[0], n_inputs, 0]
        chain_rule3 = W[n, bound[0], 1] \
            * activationFunction([suma], [0, 1]) * Z[bound[1], 0]
    elif case == [1, 2]:
        suma = np.dot(Z[0:n_inputs, 0], W[bound[0], 0:n_inputs, 0]) \
               + W[bound[0], n_inputs, 0]
        chain_rule3 = W[n, bound[0], 1] * activationFunction([suma], [0, 1])

    return -chain_rule1*chain_rule2*chain_rule3
