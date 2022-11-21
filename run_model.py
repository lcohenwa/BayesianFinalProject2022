from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import arviz as az

TRAIN_PATH = os.path.join(os.getcwd(), 'training.csv')
TEST_PATH = os.path.join(os.getcwd(), 'testing.csv')
STAN_PATH = os.path.join(os.getcwd(), 'model_test.stan')


def run_training(train_path, test_path, stan_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    covariate_cols = ['daxslope']
    data_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']

    k = len(data_cols)
    n_train = len(train_data)
    n_test = len(test_data)
    d = len(covariate_cols)
    x_train = train_data[covariate_cols].to_numpy()
    y_train = train_data[data_cols].to_numpy()
    x_test = test_data[covariate_cols].to_numpy()
    data = {'K': k, 'N_test': n_test, 'N_train': n_train, 'D': d, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'do_prior_predictive': 0}

    model = CmdStanModel(stan_file=stan_path)
    fit = model.sample(data=data)
    print(fit)
    summary = fit.summary()
    summary.to_csv('summary.csv')
    print(summary)

    az.plot_trace(fit)
    plt.show()


if __name__ == '__main__':
    run_training(TRAIN_PATH, TEST_PATH, STAN_PATH)
