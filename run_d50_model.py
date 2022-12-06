from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import arviz as az

TRAIN_PATH = os.path.join(os.getcwd(), 'training.csv')
TEST_PATH = os.path.join(os.getcwd(), 'testing.csv')
STAN_PATH = os.path.join(os.getcwd(), 'd50_stan_model.stan') # test using the linear regression 
SAMPLE_PATH = os.path.join(os.getcwd(), 'stan_samples')

if os.path.exists(SAMPLE_PATH):
    for f in os.listdir(SAMPLE_PATH):
        os.remove(os.path.join(SAMPLE_PATH, f))
else:
    os.makedirs(SAMPLE_PATH)


def run_training(train_path, test_path, stan_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    covariate_cols = ['DA(sqmi)', 'slope(m/m)', 'daxslope', 'bkf_width_HAND', 'q2_width',
                    'q500_width', 'confinement', 'Percent Cohesive', 'Percent Mixed',
                    'Percent Noncohesive', 'Percent Null', 'Average Depth To Bedrock']

    data_cols = ['d50']

    n_train = len(train_data)
    n_test = len(test_data)
    d = len(covariate_cols)

    # Feed data into stan model
    x_train = train_data[covariate_cols].to_numpy()
    y_train = train_data[data_cols].to_numpy()
    x_test = test_data[covariate_cols].to_numpy()
    data = {'N_test': n_test, 'N_train': n_train, 'D': d, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test}

    model = CmdStanModel(stan_file=stan_path)
    fit = model.sample(data=data, output_dir=SAMPLE_PATH)
    print(fit)
    summary = fit.summary()
    summary.to_csv('summary.csv')
    print(summary)

    az.plot_trace(fit)
    plt.show()


if __name__ == '__main__':
    run_training(TRAIN_PATH, TEST_PATH, STAN_PATH)
