import os
import numpy as np
import pandas as pd

FILE_NAME = 'all_basins_all_data_11-2-22.csv'
IN_PATH = os.path.join(os.getcwd(), FILE_NAME)


def clean_data(path):
    # Load data
    in_data = pd.read_csv(path)

    # Make relations
    in_data['daxslope'] = in_data['DA(sqmi)'] * in_data['slope(m/m)']

    # Filter
    # Filter out all lakes, reservoirs, etc
    in_data = in_data[in_data['waterbody'] != 1]
    # Filter out all reaches where length <1.  May have led to some errors
    in_data = in_data[in_data['model_length(m)'] > 4]
    # Filter out zeros from some columns
    for col in ['PK50AEP', 'PK0_2_width', 'PK50_width', 'bkf_width_HAND', 'CHSSPQ2', 'CH_V_Q500', 'OB_V_Q500','CHSSPQ2', 'OBSSPQ2']:
        in_data = in_data[in_data[col] > 0]
    # Filter out nodata rows from average depth to bedrock
    in_data = in_data[in_data['Average Depth To Bedrock'] != -9999]

    # Transform
    # Log transform some variables
    log_transform_list = ['daxslope', 'DA(sqmi)', 'slope(m/m)', 'PK0_2_width', 'PK50_width', 'bkf_width_HAND', 'CH_V_Q500', 'OB_V_Q500','CHSSPQ2', 'OBSSPQ2']
    for col in log_transform_list:
        in_data[col] = np.log10(in_data[col])

    # Normalize some variables
    normalize_list = ['daxslope', 'DA(sqmi)', 'slope(m/m)', 'PK0_2_width', 'PK50_width', 'bkf_width_HAND', 'CH_V_Q500', 'OB_V_Q500','CHSSPQ2', 'OBSSPQ2']
    for col in normalize_list:
        mean = in_data[col].mean()
        in_data[col] = [i - mean for i in in_data[col]]

    in_data.to_csv(f'{path[:-4]}_cleaned.csv')


def clean_data_norm01(path):
    """ Cleans data, but instead of standard normal normalizing, it normalizes 0 to 1 """
    # Load data
    in_data = pd.read_csv(path)

    # Make relations
    in_data['daxslope'] = in_data['DA(sqmi)'] * in_data['slope(m/m)']

    # Filter
    # Filter out all lakes, reservoirs, etc
    in_data = in_data[in_data['waterbody'] != 1]
    # Filter out all reaches where length <1.  May have led to some errors
    in_data = in_data[in_data['model_length(m)'] > 4]
    # Filter out zeros from some columns
    for col in ['PK50AEP', 'PK0_2_width', 'PK50_width', 'bkf_width_HAND', 'CHSSPQ2', 'CH_V_Q500', 'OB_V_Q500',
                'CHSSPQ2', 'OBSSPQ2']:
        in_data = in_data[in_data[col] > 0]
    # Filter out nodata rows from average depth to bedrock
    in_data = in_data[in_data['Average Depth To Bedrock'] != -9999]

    # Transform
    # Log transform some variables
    log_transform_list = ['daxslope', 'DA(sqmi)', 'slope(m/m)', 'PK0_2_width', 'PK50_width', 'bkf_width_HAND', 'CH_V_Q500',
                          'OB_V_Q500', 'CHSSPQ2', 'OBSSPQ2']
    for col in log_transform_list:
        in_data[col] = np.log10(in_data[col])

    # Normalize some variables
    normalize_list = ['daxslope', 'DA(sqmi)', 'slope(m/m)', 'PK0_2_width', 'PK50_width', 'bkf_width_HAND', 'CH_V_Q500', 'OB_V_Q500',
                      'CHSSPQ2', 'OBSSPQ2', 'Percent Cohesive', 'Percent Mixed', 'Percent Noncohesive', 'Average Depth To Bedrock']
    for col in normalize_list:
        max = in_data[col].max()
        min = in_data[col].min()
        range_ = max - min
        in_data[col] = [(i - min) / range_ for i in in_data[col]]

    output_cols = ['Code', *normalize_list]
    in_data[output_cols].to_csv(f'{path[:-4]}_cleaned01.csv')


if __name__ == '__main__':
    clean_data(IN_PATH)
