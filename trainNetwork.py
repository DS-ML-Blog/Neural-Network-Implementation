from numpy import random
from datetime import datetime as dt

from functions import *


def train_network(n_inputs, n_hidden, n_outputs, n_training, input_limits,
                  w_limits, conv):
  
    time_start = dt.now()
    # 1. Initialization of weights matrixes (W1, W2 and W), outputs (Z),
    # preoutputs(Y) and negative gradients (G)
    W1 = random.uniform(w_limits[0][0], w_limits[0][1],
                        [n_hidden, n_inputs+1])
    W2 = random.uniform(w_limits[1][0], w_limits[1][1],
                        [n_outputs, n_hidden+1])
    W = np.zeros([max(n_hidden, n_outputs), max(n_hidden+1, n_inputs+1), 2])
    W[0:np.shape(W1)[0], 0:np.shape(W1)[1], 0] = W1
    W[0:np.shape(W2)[0], 0:np.shape(W2)[1], 1] = W2
    G = np.zeros([max(n_hidden, n_outputs), max(n_hidden+1, n_inputs+1), 2])
    Y = np.zeros([max(n_inputs, n_hidden, n_outputs), 3])
    Z = np.zeros([max(n_inputs, n_hidden, n_outputs), 3])

    # 2. Network training
    i = 0
    stop = 0
    costs = []
    residua = []
    training_set = generateTrainingSet(n_training, n_inputs, input_limits)

    while stop == 0:
        W1 = W[0:n_hidden, 0:n_inputs+1, 0]
        W2 = W[0:n_outputs, 0:n_hidden+1, 1]

        for t in range(len(training_set)):
            # Filling Y and Z matrices
            # a) Preoutputs and outputs from first layer
            y1 = training_set[t, 0:n_inputs]
            z1 = activationFunction(y1, [0, 0])
            Y[0:len(y1), 0] = y1
            Z[0:len(z1), 0] = z1

            # b) Preoutputs and outputs from second layer
            y2 = np.matmul(z1.transpose(), W1[:, :-1].transpose()) + W1[:, -1]
            z2 = activationFunction(y2, [0, 1])
            Y[0:len(y2), 1] = y2
            Z[0:len(z2), 1] = z2

            # c) Preoutputs and outputs from third layer
            y3 = np.matmul(z2.transpose(), W2[:, :-1].transpose()) + W2[:, -1]
            z3 = activationFunction(y3, [0, 2])
            Y[0:len(y3), 2] = y3
            Z[0:len(z3), 2] = z3

            for n in range(n_outputs):

                for p in range(np.shape(W)[0]):
                    for q in range(np.shape(W)[1]):
                        for l in range(np.shape(W)[2]):

                            if W[p, q, l] != 0:

                                if l == 0:
                                    if q == n_inputs:
                                        case = [1, 2]
                                    else:
                                        case = [0, 2]
                                else:
                                    if n == p:
                                        if q == n_hidden:
                                            case = [1, 1]
                                        else:
                                            case = [0, 1]
                                    else:
                                        if q == n_hidden:
                                            case = [1, 0]
                                        else:
                                            case = [0, 0]

                                y_n = training_set[t, n_inputs + n]
                                bound = [p, q]

                                G[p, q, l] += calcGradElement(n, Z, W, y_n,
                                                              Y, case, bound,
                                                              n_inputs)

        G = G/len(training_set)
        cost = calcCost(training_set, W, n_inputs, n_hidden, n_outputs)
        costs.append(cost)

        if i >= 1:
            residuum = np.abs((costs[i]-costs[i-1])/costs[i])
            residua.append(residuum)
            print('Koszt: [', str(i), '] ', cost, '   res: ', residuum)
        else:
            residuum = 0.1
            residua.append(residuum)
            print('Koszt: [', str(i), '] ', cost)

        eta = 0.001*cost/costs[0]
        W = W + eta*G

        if (i > 200) or ((residuum < conv) and (i > 600)):  # | (costs[i] <
            # 0.1):
        # if (i>8):
        #if (i>4000) | (costs[i] < 0.1):
            stop = 1

        i += 1

    time_stop = dt.now()
    time = (time_stop - time_start).seconds

    return W, costs, residua, time
