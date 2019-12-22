import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def post_processing(W, costs_res_data, time_matrix, n_hidden_list,
                    n_training_list, deviations_matrix):

    axes = []
    vert_size = np.shape(costs_res_data)[2]
    horiz_size = np.shape(costs_res_data)[3]
    y_limits_k = [0, 1.05*costs_res_data[0, 1:, :, :].max()]
    y_limits_r = [0, 1.05*costs_res_data[1, 1:, :, :].max()]

    x_max = 1
    for i in range(vert_size):
        for j in range(horiz_size):
            if len(costs_res_data[0, 1:, i, j]
                   [costs_res_data[0, 1:, i, j] != 0]) > x_max:
                x_max = len(costs_res_data[0, 1:, i, j]
                            [costs_res_data[0, 1:, i, j] != 0])

    iterator = 1
    fig = plt.subplots(vert_size, horiz_size, dpi=80, figsize=(18, 10),
                       facecolor='white')

    for i in range(vert_size):
        for j in range(horiz_size):

            # costs
            axes.append(plt.subplot(vert_size, horiz_size, iterator))
            non_zero_values \
                = costs_res_data[0, 1:, i, j][costs_res_data[0, 1:, i, j] > 0]
            data_to_plot \
                = costs_res_data[0, 1:, i, j][costs_res_data[0, 1:, i, j] != 0]
            plt.plot(data_to_plot, color='red')
            axes[-1].set_ylabel('Cost', color='red', fontsize=16)
            axes[-1].tick_params('y', colors='red')
            plt.xlabel('Iterations', fontsize=12)
            title = 'Time: ' + str(round(time_matrix[i, j], 0)) \
                    + 's        n_H: ' + str(round(n_hidden_list[i], 0)) \
                    + '        n_T: ' + str(n_training_list[j])
            plt.title(title)
            plt.xlim([0, x_max])
            plt.ylim(y_limits_k)

            # residua
            axes.append(axes[len(axes)-1].twinx())
            non_zero_values \
                = costs_res_data[1, 1:, i, j][costs_res_data[0, 1:, i, j] > 0]
            data_to_plot \
                = costs_res_data[1, 1:, i, j][costs_res_data[1, 1:, i, j] != 0]
            plt.plot(data_to_plot, color='blue')
            axes[-1].set_ylabel('Residues', color='blue', fontsize=16)
            axes[-1].tick_params('y', colors='blue')
            plt.xlim([0, x_max])
            plt.ylim(y_limits_r)

            iterator += 1

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.45, hspace=0.45)
    plt.show()

    plt.savefig(figure=fig, fname='postpro.png')

    # -----
    fig = plt.figure()
    sns.heatmap(time_matrix, xticklabels=n_training_list,
                yticklabels=n_hidden_list, cmap='jet')
    plt.title('Czas działania algorytmu uczącego\n dla poszczególnych '
              'kombinacji n_H i n_T', fontsize=16)
    plt.xlabel('Liczba przykładów uczących', fontsize=14)
    plt.ylabel('Liczba neuronów ukrytych', fontsize=14)

    plt.savefig(figure = fig, fname = 'czas_heatmap.png')

    # -----
    fig = plt.figure()
    sns.heatmap(deviations_matrix, xticklabels=n_training_list,
                yticklabels=n_hidden_list, cmap='jet')
    plt.title('Macierz względnych odchyłek\n dla poszczególnych kombinacji '
              'n_H i n_T', fontsize=16)
    plt.xlabel('Liczba przykładów uczących', fontsize=14)
    plt.ylabel('Liczba neuronów ukrytych', fontsize=14)

    plt.savefig(figure=fig, fname='macierz_odchyłek.png')
