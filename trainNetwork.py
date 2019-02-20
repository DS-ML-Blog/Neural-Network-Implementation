import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from funkcje import *
from datetime import datetime as dt

def trainNetwork(n_inputs, n_hidden, n_outputs, n_training, input_limits, w_limits, conv):
    '''
    input_limits - macierz gdzie kolumny odpowiadają kolejnym inputom, wiersze - górny i dolny limit wartości
    w_limits - analogiczna macierz limitów wag w kolejnych warstwach
    conv - tutaj zawierają się informacje o warunku zbieżności
    '''
    czas_start = dt.now()
    #  1. Inicjalizacja macierzy wag: W1, W2 i W, aktywacji A, preoutputów Z oraz ujemnych gradientów G
    W1 = random.uniform(w_limits[0][0],w_limits[0][1],[n_hidden,n_inputs+1])
    W2 = random.uniform(w_limits[1][0],w_limits[1][1],[n_outputs,n_hidden+1])
    W = np.zeros([max(n_hidden, n_outputs), max(n_hidden+1, n_inputs+1),2])     # 3D macierz wag
    W[0:np.shape(W1)[0],0:np.shape(W1)[1],0] = W1
    W[0:np.shape(W2)[0],0:np.shape(W2)[1],1] = W2
    G = np.zeros([max(n_hidden, n_outputs), max(n_hidden+1, n_inputs+1),2])     # macierz gradientów
    Z = np.zeros([max(n_inputs, n_hidden, n_outputs), 3])                       # macierz aktywacji
    Y = np.zeros([max(n_inputs, n_hidden, n_outputs), 3])                       # macierz preoutputów

    # 2. Nauka sieci
    i = 0
    stop = 0
    koszty = []
    residua = []
    training_set = generateTrainingSet(n_training, n_inputs, input_limits)

    while stop == 0:
        W1 = W[0:n_hidden,0:n_inputs+1,0]
        W2 = W[0:n_outputs,0:n_hidden+1,1]

        for t in range(len(training_set)):                   # for1: dla wszystkich przykładów treningowych
            # Uzupełnienie wartości macierzy Z i A
            # a) Preoutputy i aktywacje pierwszej warstwy
            y1 = training_set[t,0:n_inputs]
            z1 = activationFunction(y1,[0,0])
            Y[0:len(y1),0] = y1
            Z[0:len(z1),0] = z1

            # b) Preoutputy i aktywacje drugiej warstwy
            y2 = np.matmul(z1.transpose(),W1[:,:-1].transpose()) + W1[:,-1]
            z2 = activationFunction(y2,[0,1])
            Y[0:len(y2),1] = y2
            Z[0:len(z2),1] = z2

            # c) Preoutputy i aktywacje trzeciej warstwy
            y3 = np.matmul(z2.transpose(),W2[:,:-1].transpose()) + W2[:,-1]
            z3 = activationFunction(y3,[0,2])
            Y[0:len(y3),2] = y3
            Z[0:len(z3),2] = z3

            for n in range(n_outputs):                      # for2: dla wszystkich neuronów wyjściowych

                for p in range(np.shape(W)[0]):             # for3:
                    for q in range(np.shape(W)[1]):         # for4:
                        for l in range(np.shape(W)[2]):     # for5:

                            #  jeżeli dana waga w macierzy wag 'istnieje' (jest niezerowa)
                            if W[p,q,l] != 0:
                                g = 0
                                ''' case = ...
                                [0,0] - waga neuronu innego niż aktualnie rozpatrywany
                                [0,1] - waga neuronu aktualnie rozpatrywanego
                                [0,2] - dowolna waga z pierwszej (lewej) warstwy wag
                                [1,0] - bias neuronu innego niż aktualnie rozpatrywany
                                [1,1] - bias neuronu aktualnie rozpatrywanego
                                [1,2] - dowolny bias z pierwszej (lewej) warstwy wag
                                '''
                                if l == 0:
                                    if q == n_inputs:
                                        case = [1,2]
                                    else:
                                        case = [0,2]
                                else:
                                    if n == p:
                                        if q == n_hidden:
                                            case = [1,1]
                                        else:
                                            case = [0,1]
                                    else:
                                        if q == n_hidden:
                                            case = [1,0]
                                        else:
                                            case = [0,0]

                                # 2. Następnie przypisanie wartości y_n (oczekiwany output)...
                                # ... i bound (wektor indeksów dwóch neuronów w danym połączeniu)
                                y_n = training_set[t,n_inputs + n]
                                bound = [p,q]
                                # 3. Pojedynczy element macierzy gradientu
                                G[p,q,l] += calcGradElement(n, Z, W, y_n, Y, case, bound, n_inputs)

        G = G/len(training_set)
        koszt = calcCost(training_set, W, n_inputs, n_hidden, n_outputs)
        koszty.append(koszt)

        if i>=1:
            residuum = np.abs( (koszty[i]-koszty[i-1])/koszty[i] )
            residua.append(residuum)
            print('Koszt: [',str(i),'] ',koszt,'   res: ', residuum)
        else:
            residuum = 0.1
            residua.append(residuum)
            print('Koszt: [',str(i),'] ',koszt)


        eta = 0.001*koszt/koszty[0]
        W = W + eta*G

        #if (i>1000) | ( (residuum < conv) & (i>600)): # | (koszty[i] < 0.1):
        if (i>8):
        #if (i>4000) | (koszty[i] < 0.1):
            stop = 1

        i+=1

    czas_stop = dt.now()
    czas = (czas_stop - czas_start).seconds

    return W, koszty, residua, czas

    #
