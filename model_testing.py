from cmdstanpy import CmdStanModel
import numpy as np
import os
import matplotlib.pyplot as plt


def dummy_data():
    stan_file = os.path.join(os.getcwd(), 'model_test_2.stan')

    model = CmdStanModel(stan_file=stan_file)
    k = 7
    n = 100
    d = 1
    x = np.ones(int(n / 2))
    x = np.append(x, x * 2)[:, np.newaxis]
    y1 = np.tile([50, 50, 0, 0, 0, 0, 0], (int(n / 2), 1))
    y2 = np.tile([0, 0, 0, 0, 0, 50, 50], (int(n / 2), 1))
    y = np.append(y1, y2, axis=0).astype(int)

    data = {'K': k, 'N': n, 'D': d, 'x': x, 'y': y}

    fit = model.sample(data=data)
    summary = fit.summary()

    summary['expits'] = 1/(1 + np.exp(-summary['Mean']))
    bins = list(range(1, k + 1))

    fig, ax = plt.subplots()
    for i in [1, 2]:
        adj_points = summary['Mean'][2:] - (i * summary['Mean'][1])
        print(i)
        pt_list = list()
        for j in adj_points:
            print(1 / (1 + np.exp(-j)))
            pt_list.append(1 / (1 + np.exp(-j)))
        pt_list.append(1)
        ax.plot(bins, pt_list, label=i)

    plt.legend()
    ax.set_xlabel('Bin')
    ax.set_ylabel('Cumulative Probability')
    plt.show()

dummy_data()
