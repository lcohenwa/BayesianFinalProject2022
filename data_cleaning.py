import numpy as np
import pandas as pd
import math


def clean_probhand_data(reach_path, hydraulic_path):
    reach_data = pd.read_csv(reach_path)
    hydraulic_data = pd.read_csv(hydraulic_path)

    reach_data = reach_data[['Code', 'DA(sqmi)', 'slope(m/m)', 'model_length(m)', 'bkf_width_HAND', 'Percent Cohesive', 'Percent Mixed', 'Percent Noncohesive', 'Percent Null', 'Average Depth To Bedrock']]

    reach_data = pd.merge(reach_data, hydraulic_data, how='left', left_on='Code', right_on='REACH')

    reach_data = reach_data[reach_data['LENGTH'] > 1]
    reach_data = reach_data[~reach_data['2_SA'].isna()]

    reach_data['DA(sqmi)'] = reach_data['DA(sqmi)'].apply(lambda x: 2 if x < 2 else x)

    reach_data['daxslope'] = reach_data['DA(sqmi)'] * reach_data['slope(m/m)']  # Checked and this has to occur before the log transforms or the relationship doesnt hold
    reach_data['daxslope'] = np.log(reach_data['daxslope'])
    reach_data['daxslope'] = (reach_data['daxslope'] - min(reach_data['daxslope'])) / (
            max(reach_data['daxslope']) - min(reach_data['daxslope']))

    reach_data['DA(sqmi)'] = np.log(reach_data['DA(sqmi)'])
    reach_data['DA(sqmi)'] = (reach_data['DA(sqmi)'] - np.log(2)) / (np.log(1500) - np.log(2))  # Normalize on 0-1 for all DAs in the basin

    reach_data['slope(m/m)'] = np.log(reach_data['slope(m/m)'])
    reach_data['slope(m/m)'] = -(reach_data['slope(m/m)'] - np.log(0.5)) / (np.log(0.5) - np.log(0.00001))

    reach_data['bkf_width_HAND'] = reach_data['bkf_width_HAND'].apply(lambda x: 1 if x < 1 else x)
    reach_data['bkf_width_HAND'] = np.log(reach_data['bkf_width_HAND'])
    reach_data['bkf_width_HAND'] = reach_data['bkf_width_HAND'] / max(reach_data['bkf_width_HAND'])
    print(max(reach_data['bkf_width_HAND']))

    # Percents are already normalized

    avg_avg_depth_to_bed = reach_data[~reach_data['Average Depth To Bedrock'].isin([999, -9999])]['Average Depth To Bedrock'].mean()
    max_depth = reach_data[~reach_data['Average Depth To Bedrock'].isin([999, -9999])]['Average Depth To Bedrock'].max()
    reach_data['Average Depth To Bedrock'] = reach_data['Average Depth To Bedrock'].apply(
        lambda x: max_depth if x == 999 else x)
    reach_data['Average Depth To Bedrock'] = reach_data['Average Depth To Bedrock'].apply(
        lambda x: avg_avg_depth_to_bed if x == -9999 else x)

    reach_data['Average Depth To Bedrock'] = reach_data['Average Depth To Bedrock'] / max_depth

    reach_data['q2_width'] = reach_data['2_SA'] / reach_data['LENGTH']
    reach_data['q2_width'] = reach_data['q2_width'].apply(lambda x: 1 if x < 1 else x)

    reach_data['q500_width'] = reach_data['500_SA'] / reach_data['LENGTH']
    reach_data['q500_width'] = reach_data['q500_width'].apply(lambda x: 1 if x < 1 else x)

    reach_data['confinement'] = reach_data['q500_width'] / reach_data['q2_width']

    reach_data['q2_width'] = np.log(reach_data['q2_width'])
    reach_data['q500_width'] = np.log(reach_data['q500_width'])
    reach_data['confinement'] = np.log(reach_data['confinement'])

    reach_data['q2_width'] = (reach_data['q2_width'] - min(reach_data['q2_width'])) / (
                max(reach_data['q2_width']) - min(reach_data['q2_width']))
    reach_data['q500_width'] = (reach_data['q500_width'] - min(reach_data['q500_width'])) / (
                max(reach_data['q500_width']) - min(reach_data['q500_width']))
    reach_data['confinement'] = (reach_data['confinement'] - min(reach_data['confinement'])) / (
                max(reach_data['confinement']) - min(reach_data['confinement']))

    return reach_data[['Code', 'DA(sqmi)', 'slope(m/m)', 'daxslope', 'bkf_width_HAND', 'q2_width', 'q500_width', 'confinement', 'Percent Cohesive', 'Percent Mixed', 'Percent Noncohesive', 'Percent Null', 'Average Depth To Bedrock']]

def clean_probhand_data_stats(reach_path, hydraulic_path):
    reach_data = pd.read_csv(reach_path)
    hydraulic_data = pd.read_csv(hydraulic_path)

    reach_data = reach_data[['Code', 'DA(sqmi)', 'slope(m/m)', 'model_length(m)', 'bkf_width_HAND', 'Percent Cohesive', 'Percent Mixed', 'Percent Noncohesive', 'Percent Null', 'Average Depth To Bedrock']]

    reach_data = pd.merge(reach_data, hydraulic_data, how='left', left_on='Code', right_on='REACH')

    reach_data = reach_data[reach_data['LENGTH'] > 1]
    reach_data = reach_data[~reach_data['2_SA'].isna()]

    reach_data['DA(sqmi)'] = reach_data['DA(sqmi)'].apply(lambda x: 2 if x < 2 else x)

    reach_data['daxslope'] = reach_data['DA(sqmi)'] * reach_data['slope(m/m)']  # Checked and this has to occur before the log transforms or the relationship doesnt hold
    reach_data['daxslope'] = np.log(reach_data['daxslope'])
    reach_data['daxslope'] = (reach_data['daxslope'] - reach_data['daxslope'].mean()) / reach_data['daxslope'].std()

    reach_data['DA(sqmi)'] = np.log(reach_data['DA(sqmi)'])
    reach_data['DA(sqmi)'] = (reach_data['DA(sqmi)'] - reach_data['DA(sqmi)'].mean()) / reach_data['DA(sqmi)'].std()

    reach_data['slope(m/m)'] = np.log(reach_data['slope(m/m)'])
    reach_data['slope(m/m)'] = (reach_data['slope(m/m)'] - reach_data['slope(m/m)'].mean()) / reach_data['slope(m/m)'].std()

    reach_data['bkf_width_HAND'] = reach_data['bkf_width_HAND'].apply(lambda x: 1 if x < 1 else x)
    reach_data['bkf_width_HAND'] = np.log(reach_data['bkf_width_HAND'])
    reach_data['bkf_width_HAND'] = (reach_data['bkf_width_HAND'] - reach_data['bkf_width_HAND'].mean()) / reach_data['bkf_width_HAND'].std()

    # Percents are already normalized

    avg_avg_depth_to_bed = reach_data[~reach_data['Average Depth To Bedrock'].isin([999, -9999])]['Average Depth To Bedrock'].mean()
    max_depth = reach_data[~reach_data['Average Depth To Bedrock'].isin([999, -9999])]['Average Depth To Bedrock'].max()
    reach_data['Average Depth To Bedrock'] = reach_data['Average Depth To Bedrock'].apply(
        lambda x: max_depth if x == 999 else x)
    reach_data['Average Depth To Bedrock'] = reach_data['Average Depth To Bedrock'].apply(
        lambda x: avg_avg_depth_to_bed if x == -9999 else x)

    reach_data['Average Depth To Bedrock'] = (reach_data['Average Depth To Bedrock'] - reach_data['Average Depth To Bedrock'].mean()) / reach_data['Average Depth To Bedrock'].std()

    reach_data['q2_width'] = reach_data['2_SA'] / reach_data['LENGTH']
    reach_data['q2_width'] = reach_data['q2_width'].apply(lambda x: 1 if x < 1 else x)

    reach_data['q500_width'] = reach_data['500_SA'] / reach_data['LENGTH']
    reach_data['q500_width'] = reach_data['q500_width'].apply(lambda x: 1 if x < 1 else x)

    reach_data['confinement'] = reach_data['q500_width'] / reach_data['q2_width']

    reach_data['q2_width'] = np.log(reach_data['q2_width'])
    reach_data['q500_width'] = np.log(reach_data['q500_width'])
    reach_data['confinement'] = np.log(reach_data['confinement'])

    reach_data['q2_width'] = (reach_data['q2_width'] - reach_data['q2_width'].mean()) / reach_data['q2_width'].std()
    reach_data['q500_width'] = (reach_data['q500_width'] - reach_data['q500_width'].mean()) / reach_data['q500_width'].std()
    reach_data['confinement'] = (reach_data['confinement'] - reach_data['confinement'].mean()) / reach_data['confinement'].std()

    return reach_data[['Code', 'DA(sqmi)', 'slope(m/m)', 'daxslope', 'bkf_width_HAND', 'q2_width', 'q500_width', 'confinement', 'Percent Cohesive', 'Percent Mixed', 'Percent Noncohesive', 'Percent Null', 'Average Depth To Bedrock']]


def merge(clean_reach_path, gsd_path):
    reach_data = pd.read_csv(clean_reach_path)
    gsd_data = pd.read_csv(gsd_path)

    gsd_data = gsd_data[gsd_data['Sand'] != -9999]

    merged = pd.merge(gsd_data, reach_data, how='inner', left_on='UVM_ID', right_on='Code')
    return merged

def split_training_testing(merged_path, split_percent=0.8):
    merged_dataset = pd.read_csv(merged_path)
    max_min_rows = list()
    cols = [c for c in merged_dataset.columns if c not in ['Code', 'UVM_ID', 'SGAT_ID', 'd50', 'd84', 'consistenc']]
    for col in cols:
        max_min_rows.append(np.argmin(merged_dataset[col]))
        max_min_rows.append(np.argmax(merged_dataset[col]))
    max_min_rows = set(max_min_rows)

    leftover_rows = set(range(len(merged_dataset))).difference(max_min_rows)
    split_count = math.floor((len(merged_dataset) * split_percent) - len(max_min_rows))
    add_to_training = set(np.random.choice(list(leftover_rows), split_count, replace=False))

    training = max_min_rows.union(add_to_training)
    testing = leftover_rows.difference(add_to_training)

    training = merged_dataset.iloc[list(training)]
    testing = merged_dataset.iloc[list(testing)]

    return testing, training

def clean_data(stats=False):
    reach_path = r"D:\Academic\UVM\GSDs\scott_working\11-13-22\all_basins_all_data_11-10-22.csv"
    hydraulic_path = r"D:\Academic\UVM\GSDs\scott_working\11-13-22\all_interpolated_hydraulics_optimized.csv"
    gsd_path = r"D:\Academic\UVM\GSDs\scott_working\11-13-22\gsd_dataset_11-3-22.csv"

    out_probhand_path = r"D:\Academic\UVM\GSDs\scott_working\11-13-22\cleaned_transformed_full_network.csv"
    out_merged_path = r"D:\Academic\UVM\GSDs\scott_working\11-13-22\cleaned_merged.csv"

    testing_path = r"C:\Users\Kensf\PycharmProjects\Geospatial_Hydrology\bayesian_stats\project\BayesianFinalProject2022\testing.csv"
    training_path = r"C:\Users\Kensf\PycharmProjects\Geospatial_Hydrology\bayesian_stats\project\BayesianFinalProject2022\training.csv"

    if stats:
        cleaned_probhand = clean_probhand_data_stats(reach_path, hydraulic_path)
    else:
        cleaned_probhand = clean_probhand_data(reach_path, hydraulic_path)
    for col in cleaned_probhand.columns:
        print(col)
        print(cleaned_probhand.describe()[col])
    cleaned_probhand.to_csv(out_probhand_path, index=False)

    merged = merge(out_probhand_path, gsd_path)
    merged.to_csv(out_merged_path, index=False)

    test, train = split_training_testing(out_merged_path)
    test.to_csv(testing_path, index=False)
    train.to_csv(training_path, index=False)


if __name__ == '__main__':
    clean_data(stats=True)
