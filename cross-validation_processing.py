import cmdstanpy
import numpy as np
import pandas as pd
import os

TESTING_PATH = os.path.join(os.getcwd(), 'testing.csv')
#STAN_PATH = r'C:\Users\Kensf\PycharmProjects\Geospatial_Hydrology\bayesian_stats\project\BayesianFinalProject2022\model_testo3938gvt'
STAN_PATH = os.path.join(os.getcwd(), 'stan_samples_no_m')

model = cmdstanpy.from_csv(STAN_PATH)
draws = model.draws_pd()
cols = [c for c in draws.columns if 'y_test' in c]

summary_histograms = list()
bins = list(range(1, 9))
for c in cols:
    tmp_col = draws[c].to_numpy()
    tmp_hist = np.histogram(tmp_col, bins=bins, normed=True)
    summary_histograms.append(np.cumsum(tmp_hist[0]))

summary_histograms = np.array(summary_histograms) * 100

testing_df = pd.read_csv(TESTING_PATH)
for b in range(7):
    testing_df[f'pred{b}'] = summary_histograms[:, b]

testing_df.to_csv(os.path.join(os.getcwd(), 'testing_results_no_m.csv'))