import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_PATH = r"C:\Users\Kensf\PycharmProjects\Geospatial_Hydrology\bayesian_stats\project\BayesianFinalProject2022\testing_results.csv"
NHD_PATH = r"G:\My Drive\Academic\UVM\Courses\Fall 2022\CE359\project\11-20-22\NHD_names.csv"
OUT_PATH = r'G:\My Drive\Academic\UVM\Courses\Fall 2022\STAT330\project\11-28-22'

def plot_rmse_by_bin(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']

    rmse = df[measured_cols].to_numpy() - df[predicted_cols].to_numpy()
    rmse = np.square(rmse)
    rmse = np.sum(rmse, axis=0) / rmse.shape[0]
    print(f'Total RMSE = {np.sqrt(np.sum(rmse) / len(rmse))}')
    rmse = np.sqrt(rmse)

    fig, ax = plt.subplots()
    indices = list(range(len(rmse)))
    ax.bar(indices, rmse)
    ax.set_xticks(indices, labels=measured_cols, fontsize=8)
    ax.set_ylabel('RMSE (%)')

def calculate_rmse_scores(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']
    bin_sizes = [0.062, 2, 5.6, 16, 64, 256, 1000]

    predictions = df[predicted_cols].to_numpy()
    measured = df[measured_cols].to_numpy()
    d50_pred = np.array([np.interp(50, predictions[i], bin_sizes) for i in range(predictions.shape[0])])
    d50_measured = np.array([np.interp(50, measured[i], bin_sizes) for i in range(measured.shape[0])])

    rmse = d50_measured - d50_pred
    rmse = np.square(rmse)
    rmse = np.sum(rmse) / rmse.shape[0]
    rmse = np.sqrt(rmse)
    print(f'RMSE = {rmse} (mm)')


def plot_rmse_by_da(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']

    rmse = df[measured_cols].to_numpy() - df[predicted_cols].to_numpy()
    rmse = np.square(rmse)
    rmse = np.sum(rmse, axis=1) / rmse.shape[1]
    rmse = np.sqrt(rmse)

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(df['DA(sqmi)'], rmse, fc='none', ec='k', s=8, alpha=0.5)
    axs[0].set_ylabel('RMSE (%)')
    axs[0].set_xlabel(r'DA ($mi^2$)')

    axs[1].scatter(df['slope(m/m)'], rmse, fc='none', ec='k', s=8, alpha=0.5)
    axs[1].set_ylabel('RMSE (%)')
    axs[1].set_xlabel(r'Slope (${m}/{m}$)')


def plot_rmse_winners(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']
    bin_list = [0.062, 2, 5.6, 16, 64, 256, 1000]

    rmse = df[measured_cols].to_numpy() - df[predicted_cols].to_numpy()
    rmse = np.square(rmse)
    rmse = np.sum(rmse, axis=1) / rmse.shape[1]
    rmse = np.sqrt(rmse)

    rmse_sorted = np.argsort(rmse)

    # filter out some names we don't like
    ignore_list = ['Jones Brook', 'Indian Brook']
    winner_ind = list()
    winner_names = list()
    count = 0
    while len(winner_ind) < 4:
        tmp_winner = rmse_sorted[count]
        tmp_winner_name = df.iloc[tmp_winner]['gnis_name']
        if tmp_winner_name in winner_names or tmp_winner_name in ignore_list or pd.isna(tmp_winner_name):
            count += 1
            continue
        else:
            winner_names.append(tmp_winner_name)
            winner_ind.append(tmp_winner)
            count += 1

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for ind, row in enumerate(winner_ind):
        if ind == 0:
            i = 0
            j = 0
        elif ind == 1:
            i = 1
            j = 0
        elif ind == 2:
            i = 1
            j = 1
        elif ind == 3:
            i = 0
            j = 1
        axs[i, j].plot(bin_list, df.iloc[row][measured_cols], c='k', label='measured')
        axs[i, j].plot(bin_list, df.iloc[row][predicted_cols], c='r', ls='dashed', label='predicted')
        axs[i, j].text(min(bin_list), 100, f'RMSE={round(rmse[row], 2)}', va='top', ha='left')
        axs[i, j].set_xscale('log')
        axs[i, j].set_title(df.iloc[row]['gnis_name'])
    axs[0, 0].legend(loc='lower right')
    axs[1, 0].set_ylabel('Percent Finer')
    axs[0, 0].set_ylabel('Percent Finer')
    axs[1, 1].set_xlabel('Grain Size (mm)')
    axs[1, 0].set_xlabel('Grain Size (mm)')
    fig.suptitle('Winners')

def plot_rmse_losers(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']
    bin_list = [0.062, 2, 5.6, 16, 64, 256, 1000]

    rmse = df[measured_cols].to_numpy() - df[predicted_cols].to_numpy()
    rmse = np.square(rmse)
    rmse = np.sum(rmse, axis=1) / rmse.shape[1]
    rmse = np.sqrt(rmse)

    rmse_sorted = np.argsort(rmse)

    # filter out any duplicate names
    winner_ind = list()
    winner_names = list()
    count = 1
    while len(winner_ind) < 4:
        tmp_winner = rmse_sorted[-count]
        tmp_winner_name = df.iloc[tmp_winner]['gnis_name']
        if tmp_winner_name in winner_names or pd.isna(tmp_winner_name):
            count += 1
            continue
        else:
            winner_names.append(tmp_winner_name)
            winner_ind.append(tmp_winner)
            count += 1

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for ind, row in enumerate(winner_ind):
        if ind == 0:
            i = 0
            j = 0
        elif ind == 1:
            i = 1
            j = 0
        elif ind == 2:
            i = 1
            j = 1
        elif ind == 3:
            i = 0
            j = 1
        axs[i, j].plot(bin_list, df.iloc[row][measured_cols], c='k', label='measured')
        axs[i, j].plot(bin_list, df.iloc[row][predicted_cols], c='r', ls='dashed', label='predicted')
        axs[i, j].text(min(bin_list), 100, f'RMSE={round(rmse[row], 2)}', va='top', ha='left')
        axs[i, j].set_xscale('log')
        axs[i, j].set_title(df.iloc[row]['gnis_name'])
    axs[0, 0].legend(loc='lower right')
    axs[1, 0].set_ylabel('Percent Finer')
    axs[0, 0].set_ylabel('Percent Finer')
    axs[1, 1].set_xlabel('Grain Size (mm)')
    axs[1, 0].set_xlabel('Grain Size (mm)')
    fig.suptitle('Losers')


def plot_rmse_familiar_favorites(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']
    bin_list = [0.062, 2, 5.6, 16, 64, 256, 1000]

    rmse = df[measured_cols].to_numpy() - df[predicted_cols].to_numpy()
    rmse = np.square(rmse)
    rmse = np.sum(rmse, axis=1) / rmse.shape[1]
    rmse = np.sqrt(rmse)

    favorites_list = [7000250, 10011680, 3000159, 8000967]
    # 7000695 Balck Creek
    # 3002898 Lewis Creek at N Ferrisburg
    # 3002898 Dog Creek at Northfield
    # 3000963 Dog River at West Berlin
    favorite_names = ['Black Creek', 'Castleton River d/s Castleton', 'Mad River d/s Pine Brook', 'Lewis Creek at Hinesburg']
    rows = df[df['UVM_ID'].isin(favorites_list)]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for ind, row in enumerate(rows.iterrows()):
        if ind == 0:
            i = 0
            j = 0
        elif ind == 1:
            i = 1
            j = 0
        elif ind == 2:
            i = 1
            j = 1
        elif ind == 3:
            i = 0
            j = 1
        axs[i, j].plot(bin_list, row[1][measured_cols].values, c='k', label='measured')
        axs[i, j].plot(bin_list, row[1][predicted_cols].values, c='r', ls='dashed', label='predicted')
        axs[i, j].text(min(bin_list), 100, f'RMSE={round(rmse[row[0]], 2)}', va='top', ha='left')
        axs[i, j].set_xscale('log')
        axs[i, j].set_title(favorite_names[ind])
    axs[0, 0].legend(loc='lower right')
    axs[1, 0].set_ylabel('Percent Finer')
    axs[0, 0].set_ylabel('Percent Finer')
    axs[1, 1].set_xlabel('Grain Size (mm)')
    axs[1, 0].set_xlabel('Grain Size (mm)')
    fig.suptitle('Familiar Favorites')

def plot_gsd(df):
    measured_cols = ['SiltOrSmal', 'Sand', 'FineGravel', 'CourseGrav', 'Cobble', 'Boulder', 'Bedrock']
    predicted_cols = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']
    bin_list = [0.062, 2, 5.6, 16, 64, 256, 1000]

    predicted_cdfs = list()
    measured_cdfs = list()
    percent_100 = np.array([0 for i in measured_cols])
    for i in df[predicted_cols].iterrows():
        predicted_cdfs.append(i[1].to_numpy())
        hundreds = (i[1] >= 100) * 1
        percent_100 = percent_100 + hundreds
    for i in df[measured_cols].iterrows():
        measured_cdfs.append(i[1].to_numpy())

    percent_100 = list(reversed(percent_100 / (len(df))))

    fig, ax = plt.subplots(2, 1, sharex=True)

    tmp_cols = list(reversed(['na', *measured_cols]))
    tmp_x_list = list(reversed([0.01, *bin_list]))
    for ind in range(len(tmp_cols) - 2):
        ax[0].axvline(x=tmp_x_list[ind], color='k', label=tmp_cols[ind])
        ax[1].axvline(x=tmp_x_list[ind], color='k', label=tmp_cols[ind])
        x = (10 ** (np.log10(tmp_x_list[ind]) - ((np.log10(tmp_x_list[ind]) - np.log10(tmp_x_list[ind + 1])) / 2)))
        ax[0].text(x, 0, tmp_cols[ind], va='top', ha='center', fontsize='xx-small')
        ax[1].text(x, 0, tmp_cols[ind], va='top', ha='center', fontsize='xx-small')
        #percent_text = r'$\Swarrow${}%'.format(int(percent_100[ind] * 100))
        #ax.text(tmp_x_list[ind], 100, percent_text, va='bottom', ha='left')

    for pred, meas in zip(predicted_cdfs, measured_cdfs):
        ax[0].plot(bin_list, meas, alpha=0.01, c='b')
        ax[1].plot(bin_list, pred, alpha=0.01, c='r')

    ax[0].set_title('Measured Distribution')
    ax[1].set_title('Predicted Distribution')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Grain Size (mm)')
    ax[1].set_ylabel('Percent Finer')
    ax[1].set_ylabel('Percent Finer')
    fig.set_size_inches(6.5, 10)


def run_all(in_path, nhd_path=None, out_path=None):
    in_data = pd.read_csv(in_path)
    in_data['pred6'] = 100
    if nhd_path:
        nhd_data = pd.read_csv(nhd_path)
        in_data = in_data.merge(nhd_data, on='UVM_ID', suffixes=[None, '_'], how='left')

    plot_gsd(in_data)
    calculate_rmse_scores(in_data)
    plot_rmse_by_bin(in_data)
    plot_rmse_by_da(in_data)
    plot_rmse_winners(in_data)
    plot_rmse_losers(in_data)
    plot_rmse_familiar_favorites(in_data)
    plt.show()

    # if out_path:
    #     fig_nums = plt.get_fignums()
    #     figs = [plt.figure(n) for n in fig_nums]
    #     count = 0
    #     for fig in figs:
    #         count += 1
    #         fig.savefig(os.path.join(out_path, f'{count}.png'), dpi=300)


def run_all_baseline(in_path, nhd_path=None, out_path=None):
    in_data = pd.read_csv(in_path)
    baseline_distribution = np.array([(i / 7) * 100 for i in range(1, 8)])
    baseline_distribution = np.tile(baseline_distribution, len(in_data)).reshape(len(in_data), 7)
    in_data[['pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6']] = baseline_distribution
    if nhd_path:
        nhd_data = pd.read_csv(nhd_path)
        in_data = in_data.merge(nhd_data, on='UVM_ID', suffixes=[None, '_'], how='left')

    calculate_rmse_scores(in_data)
    plot_rmse_by_bin(in_data)
    plot_rmse_by_da(in_data)
    plot_rmse_winners(in_data)
    plot_rmse_losers(in_data)
    plot_rmse_familiar_favorites(in_data)
    #plt.show()

    if out_path:
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        count = 0
        for fig in figs:
            count += 1
            fig.savefig(os.path.join(out_path, f'{count}.png'), dpi=300)


if __name__ == '__main__':
    run_all(IN_PATH, NHD_PATH, OUT_PATH)
