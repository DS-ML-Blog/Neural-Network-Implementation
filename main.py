import numpy as np
from numpy import random
from funkcje import *
from evaluateNetwork import *
from trainNetwork import trainNetwork
from postProcessing import postProcessing
from datetime import datetime as dt

# DANE WEJŚCIOWE
# -- 1. Stałe ----------------------------------
n_inputs = 4
n_outputs = 2
input_limits = [[-10,10],[-10,10],[-10,10],[-10,10]]
conv = 0.0002
input_vec = np.array([-8, 2, 1, -4])

# -- 2. Zmienne ---------------------------------
n_hidden_list = [3, 8, 15]
n_training_list = [100,500,1500]
w_limits_ini = [[-0.1,0.1],[-0.1,0.1]]
# ------------------------------------------------

# -- 3. Przygotowanie do nauki
start = dt.now()
w_limits = w_limits_ini

lista_kosztow = [[],[],[]]
lista_resid = [[],[],[]]
max_length = 0

czas_matrix = np.zeros([len(n_hidden_list),len(n_training_list)])
output_matrix = np.zeros([len(n_hidden_list), len(n_training_list), n_outputs])
deviations_matrix = np.zeros([len(n_hidden_list), len(n_training_list)])

# -- 4. Pętla ucząca - podwójny for dla n_hidden oraz n_training
for h_counter, h in enumerate(n_hidden_list):
    for t_counter, t in enumerate(n_training_list):

        W, koszty, residua, czas = trainNetwork(n_inputs, h, n_outputs, t, input_limits, w_limits, conv)
        lista_kosztow[h_counter].append(koszty)
        lista_resid[h_counter].append(residua)
        max_length = max(max_length, len(koszty))

        czas_matrix[h_counter, t_counter] = czas
        output = getNetworkOutput(input_vec, W, n_inputs, n_hidden_list[h_counter], n_outputs)
        output_matrix[h_counter, t_counter, :] = output
        deviations_matrix[h_counter, t_counter] = np.abs( (output[0] - sum(input_vec))/sum(input_vec) )

# --- 5. Zamiana listy kosztów i residuów do postaci macierzy 4D
dane_koszty_res = np.zeros([2,max_length,len(n_hidden_list),len(n_training_list)])
for h_counter, h in enumerate(n_hidden_list):
    for t_counter, t in enumerate(n_training_list):
        dane_koszty_res[0,:len(lista_kosztow[h_counter][t_counter]),h_counter, t_counter] = lista_kosztow[h_counter][t_counter]
        dane_koszty_res[1,:len(lista_resid[h_counter][t_counter]),h_counter, t_counter] = lista_resid[h_counter][t_counter]
# -------------------------------------

# -- 6. Postprocessing
stop = dt.now()
delta = stop - start

print('Czas działania całości: ', round(delta.seconds/3600, 2), ' godziny')

for h_counter, h in enumerate(n_hidden_list):
    for t_counter, t in enumerate(n_training_list):
        print('n_hidden = ', h, '\t n_training = ', t)
        print('Obliczone wartości: ' + str(round(output_matrix[h_counter, t_counter, 0], 3)) + '   ' + str(round(output_matrix[h_counter, t_counter, 1],3)) )
        print('Oczekiwane wartości: ' + str(round(input_vec[0]/9.81*2*np.sin(input_vec[1])*np.cos(input_vec[1]),3)) + '   ' \
            + str( round(input_vec[0]**2/9.81/2*(np.sin(input_vec[1]))**2,3)) + '\n' )

postProcessing(W, dane_koszty_res, czas_matrix, n_hidden_list, n_training_list, deviations_matrix)

# -- 7. Ewaluacja

r2, mse, rmse, mae, mape = evaluateNetwork(n_inputs, input_limits, W, n_hidden_list[2], n_outputs)
print('Ewaluacja: \n' + 'R2: ' + str(r2) + '\nMSE:' + str(mse) + '\nRMSE: ' + str(rmse) + '\nMAE: ' + str(mae) + '\nMAPE: ' + str(mape))

#
