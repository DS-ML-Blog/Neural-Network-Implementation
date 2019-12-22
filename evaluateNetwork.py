import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from functions import *
from sklearn.metrics import *


def evaluate_network(n_inputs, input_limits, W, n_hidden, n_outputs):

    dataset = generateTrainingSet(1000, n_inputs, input_limits)
    results_matrix = dataset[:, n_inputs:]
    predict_matrix = np.zeros(np.shape(results_matrix))

    for r, row in enumerate(predict_matrix):
        predict_matrix[r, :] = getNetworkOutput(dataset[r, :n_inputs], W,
                                                n_inputs, n_hidden, n_outputs)

    results_vector = np.matrix.flatten(results_matrix)
    predict_vector = np.matrix.flatten(predict_matrix)

    r2 = r2_score(results_vector, predict_vector)
    mse = mean_squared_error(results_vector, predict_vector)
    rmse = np.sqrt(mean_squared_error(results_vector, predict_vector))
    mae = mean_absolute_error(results_vector, predict_vector)

    zero_indices = np.where(results_vector == 0)
    results_vector = np.delete(results_vector, zero_indices)
    predict_vector = np.delete(predict_vector, zero_indices)

    mape = np.mean(np.abs((results_vector - predict_vector) /
                          results_vector)) * 100

    fig = plt.figure(dpi=1000)
    plt.scatter(x=results_vector, y=predict_vector, s=0.2)
    plt.xlim([-35, 35])
    plt.ylim([-35, 35])
    plt.title('Wartości obliczone vs. rzeczywiste', fontsize=16)
    plt.xlabel('Wartości rzeczywiste', fontsize=14)
    plt.ylabel('Wartości obliczone', fontsize=14)

    plt.savefig(figure=fig, fname='ewaluacja1.png')

    fig = plt.figure(dpi=1000)
    plt.scatter(x=results_vector, y=predict_vector, s=0.2)
    plt.plot([-30, 30], [-30, 30], 'r-', linewidth=0.8)
    plt.xlim([-35, 35])
    plt.ylim([-35, 35])
    plt.title('Wartości obliczone vs. rzeczywiste', fontsize=16)
    plt.xlabel('Wartości rzeczywiste', fontsize=14)
    plt.ylabel('Wartości obliczone', fontsize=14)

    plt.savefig(figure=fig, fname='ewaluacja2.png')

    return r2, mse, rmse, mae, mape
