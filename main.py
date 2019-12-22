import numpy as np
from datetime import datetime as dt

from evaluateNetwork import *
from trainNetwork import train_network
from postProcessing import post_processing


# INPUT DATA
# -- 1. Constants ----------------------------------
n_inputs = 4
n_outputs = 2
input_limits = [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]
conv = 0.0002
input_vec = np.array([-8, 2, 1, -4])

# -- 2. Variables ---------------------------------
n_hidden_list = [3, 8, 15]
n_training_list = [100, 500, 1500]
w_limits_ini = [[-0.1, 0.1], [-0.1, 0.1]]
# ------------------------------------------------

# -- 3. Preprocessing
start = dt.now()
w_limits = w_limits_ini

costs_list = [[], [], []]
residuals_list = [[], [], []]
max_length = 0

time_matrix = np.zeros([len(n_hidden_list), len(n_training_list)])
output_matrix = np.zeros([len(n_hidden_list), len(n_training_list), n_outputs])
deviations_matrix = np.zeros([len(n_hidden_list), len(n_training_list)])

# -- 4. Training loop - double 'for' for n_hidden and n_training
for h_counter, h in enumerate(n_hidden_list):
    for t_counter, t in enumerate(n_training_list):
        W, costs, residuals, time \
            = train_network(n_inputs, h, n_outputs, t, input_limits, w_limits,
                            conv)
        costs_list[h_counter].append(costs)
        residuals_list[h_counter].append(residuals)
        max_length = max(max_length, len(costs))

        time_matrix[h_counter, t_counter] = time
        output \
            = getNetworkOutput(input_vec, W, n_inputs,
                               n_hidden_list[h_counter], n_outputs)
        output_matrix[h_counter, t_counter, :] = output
        deviations_matrix[h_counter, t_counter] \
            = np.abs((output[0] - sum(input_vec))/sum(input_vec))

# --- 5. Transformation of costs and residuals lists to a form of 4D matrix
costs_res_data = np.zeros([2, max_length, len(n_hidden_list),
                           len(n_training_list)])
for h_counter, h in enumerate(n_hidden_list):
    for t_counter, t in enumerate(n_training_list):
        costs_res_data[0, :len(costs_list[h_counter][t_counter]), h_counter,
                       t_counter] = costs_list[h_counter][t_counter]
        costs_res_data[1, :len(residuals_list[h_counter][t_counter]),
                       h_counter, t_counter] \
            = residuals_list[h_counter][t_counter]
# -------------------------------------

# -- 6. Postprocessing
stop = dt.now()
delta = stop - start

print('Total time: ', round(delta.seconds/3600, 2), ' [h]')

for h_counter, h in enumerate(n_hidden_list):
    for t_counter, t in enumerate(n_training_list):
        print('n_hidden = ', h, '\t n_training = ', t)
        print('Calculated values: '
              + str(round(output_matrix[h_counter, t_counter, 0], 3)) + '  '
              + str(round(output_matrix[h_counter, t_counter, 1], 3)))

        exp_value_1 = sum(input_vec)
        exp_value_2 = sum(input_vec)/4

        print('Expected values: ' + str(round(exp_value_1, 3)) + '   '
              + str(round(exp_value_2, 3)) + '\n')

post_processing(W, costs_res_data, time_matrix, n_hidden_list, n_training_list,
                deviations_matrix)

# -- 7. Evaluation
r2, mse, rmse, mae, mape \
    = evaluate_network(n_inputs, input_limits, W, n_hidden_list[2], n_outputs)
print('Evaluation: \n' + 'R2: ' + str(r2) + '\nMSE:' + str(mse) + '\nRMSE: ' +
      str(rmse) + '\nMAE: ' + str(mae) + '\nMAPE: ' + str(mape))
