# -*- coding: utf-8 -*-

"""
This script is used to pre-process the forest data from the raw dataset
with the tree metric distribution for angiosperm and gymnosperm forests to a
single dataset used for the learning of tree metric distributions in function
of climatic data.

Author: Diego Bengochea 09/11/2022
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC

"""
1- Joining the angiosperm and gymnosperm datasets.
"""

# Load the shapefiles.
file_angiosperm = "/home/dibepa/Documents/TalloTreeDatabase/angiosperm_forests_tree_metric_distribution.shp"
file_gymnosperm = "/home/dibepa/Documents/TalloTreeDatabase/gymnosperm_forests_tree_metric_distribution.shp"
angiosperm_forests = gpd.read_file(file_angiosperm)
gymnosperm_forests = gpd.read_file(file_gymnosperm)

# Create a new column with information on the division and corresponding id:
# 1 for angiosperms and 2 for gymnosperms.
angiosperm_forests["div"] = "angiosperm"
gymnosperm_forests["div"] = "gymnosperm"
angiosperm_forests["div_id"] = 1
gymnosperm_forests["div_id"] = 2

# Concatenate both dataframes.
forests = pd.concat([angiosperm_forests, gymnosperm_forests]).reset_index()

"""
2- Re-scale the standard deviations with the mean value for each instance.
"""

# List of tuples with the column names for mean and standard deviation of a
# feature.
pair_columns  = [("mean_h_m","std_h_m"),("mean_cr_m","std_cr_m"),("mean_yt","std_yt"),("mean_p","std_p"),("mean_tcq","std_tcq"),("mean_tdq","std_tdq"),("mean_twq","std_twq"),("mean_pwm","std_pwm"),("mean_pet","std_pet"),("mean_ai","std_ai")]

# Iterate over the list of tuples and rescale the standard-deviation columns.
for pair in pair_columns:
    mean = pair[0]
    sd   = pair[1]
    forests[sd] = forests[sd]/forests[mean]

# The standard deviations are now expressed as fractions of the mean!

"""
3- Drop the unnecessary features: the standard deviations of the predictors.
"""

# Note that the SD of predictors serves only the purpose of checking that the
# tesselation of the world used to aggregate trees does not have great influence
# on the mean value of bioclimatic variables.

sd_columns = ["std_ai","std_pet","std_yt","std_pwm","std_tcq","std_tdq","std_twq","std_p"]
forests.drop(columns=sd_columns)

"""
4- Add P-PET column.
"""

# P-PET has been shown to be highly correlated with canopy height across forests
# of the world. TODO: find the citation from 2016.
add_p_pet = False
if add_p_pet == True:
    forests["mean_p_pet"] = forests["mean_p"] - forests["mean_pet"]

"""
5- Select uniquely the forests with more than N sampled trees
"""

N = 10.0
forests = forests[forests["n_samples"]>N]

"""
6- Identify features whose values span multiple orders of magnitude to evaluate
   the need for log-transforming them for more meaningful discretization.
"""

# Dataframe for intermediate operations.
df = gpd.GeoDataFrame(forests)

# Transform the temperatures from Celsius to Kelvin for strictly positive ranges.
temp_columns = ["mean_yt","mean_tcq","mean_tdq","mean_twq"]
for col in temp_columns:
    df[col]  = forests[col].apply(lambda x: x + 273,15)

# Normalize every numerical feature to evaluate range.
columns = ["mean_h_m","std_h_m","mean_cr_m","std_cr_m","mean_yt","mean_p","mean_tcq","mean_tdq","mean_twq","mean_pwm","mean_pet","mean_ai"]
#columns = ["mean_h_m","std_h_m","mean_cr_m","std_cr_m","mean_yt","mean_p","mean_tcq","mean_tdq","mean_twq","mean_pwm","mean_pet","mean_ai","mean_p_pet"]
max = forests[columns].agg(["max"])
for col in columns:
    df[col]= df[col]/max[col].loc["max"]

# Calculate range size as proportion of the minimum value.
min_max = df[columns].agg(["min","max"])
ranges = np.abs((min_max.loc["max"]-min_max.loc["min"])/min_max.loc["min"])

# Collect the headers where ranges are large.
range_threshold = 5.0
headers = ranges.where(lambda x: x>range_threshold).dropna().index.tolist() # this is not working as supposed
reduced_forests = forests[headers]

# Plot the pair-wise relationships between these variables to examine need for log-transformation.
show_plot = False
if show_plot == True:
    sns.set_theme(style="ticks")
    sns.pairplot(reduced_forests, corner=True,diag_kind="hist" )
    plt.show()

"""
7- Log-transform features spanning multiple orders of magnitude after inspecting
   intermediate DataFrame.
"""

# Store the target features' headers.
target_columns = ["mean_h_m","std_h_m","mean_cr_m","std_cr_m"]

# Remove the target features' headers from the features to be log-transformed
log_columns = headers
log_targets = True
if log_targets == False:
    for col in target_columns:
        log_columns.remove(col)

for col in log_columns:
    reduced_forests[col] = reduced_forests[col].apply(lambda x: np.log(x))

show_plot = False
if show_plot == True:
    sns.set_theme(style="ticks")
    sns.pairplot(reduced_forests, corner=True,diag_kind="hist" )
    plt.show()

# Log-transform the main DataFrame.
for col in log_columns:
    forests[col] = forests[col].apply(lambda x: np.log(x))

# Examine the relationship for the temperatures.
# cols = ["mean_h_m","std_h_m","mean_cr_m","std_cr_m","mean_yt","mean_tcq","mean_tdq","mean_twq","cont_id"]
# sns.set_theme(style="ticks")
# sns.pairplot(forests[cols], hue="cont_id", corner=True,diag_kind="hist" )
# plt.show()

"""
8- Balance the dataset by over-sampling with SMOTE-NC.
"""

# Specify number of bins for the target features' discretization.
n_bins = 10

# Names for the binned features columns.
binned_columns = ["hmn_bin","hsd_bin","crmn_bin","crsd_bin"]

for i,col in enumerate(target_columns):
    min    = forests[col].agg(["min"])[0]
    max    = forests[col].agg(["max"])[0]
    bins   = np.linspace(min,max,n_bins).tolist()
    labels = np.arange(n_bins).tolist()

    new_col = binned_columns[i]
    forests[new_col] = pd.cut(forests[col],bins=n_bins,labels=labels)



# Balance the data for each of the 4 target features.

# Create 4 different DataFrames with balanced data for each target.
df_hmean  = gpd.GeoDataFrame(forests)
df_hstd   = gpd.GeoDataFrame(forests)
df_crmean = gpd.GeoDataFrame(forests)
df_crstd  = gpd.GeoDataFrame(forests)

# Group the datasets in a list.
dfs = [df_hmean, df_hstd, df_crmean, df_crstd]

# Define the SMOTE-NC over-sampler.

k_neighbors = 5

# Duplicate instances with a low number of representatives in the target class
# until reaching the minimal number needed k+1 to proceed with over-sampling.
for i,col in enumerate(binned_columns):
    df = dfs[i]
    count = df[col].value_counts()
    low_count = count.where(count<k_neighbors+1)
    cat_index = low_count.dropna().index.codes
    for ind in cat_index:
        while df[df[col]==ind].shape[0] < k_neighbors+1:
            print(df[df[col]==ind].shape[0])
            df = df.append([df[df[col] == ind]], ignore_index=True)
    dfs[i] = df

# Specify the predictors to be used and which ones are categorical.
predictors   = ["mean_yt","mean_p","mean_tcq","mean_tdq","mean_twq","mean_pwm","mean_pet","mean_ai","cont_id","biome_id"]

balanced_dfs = []
# Perform the balancing.
for i,col in enumerate(binned_columns):
    target = dfs[i][col]
    predictor_df = dfs[i][predictors]

    # Get the categorical features' indices to specify the SMOTE-NC over-sampler.
    categorical_indices  = predictor_df.columns.get_indexer(['cont_id', 'biome_id'])
    # Instantiate over-sampler.
    over_sampler = SMOTENC(categorical_indices, k_neighbors=k_neighbors)

    predictors_balanced, target_balanced = over_sampler.fit_resample(predictor_df, target)
    balanced_df = pd.concat([target_balanced,predictors_balanced], axis=1)
    balanced_dfs.append(balanced_df)

"""
9- Perform feature selection to reduce degrees of freedom.
"""

# Transform the temperatures back to Celsius
for i,df in enumerate(balanced_dfs):
    for col in temp_columns:
        df[col]  = df[col].apply(lambda x: x - 273,15)
        balanced_dfs[i] = df

# Before designing a full method for feature selection it became apparent that
# PET and AI are the best predictors with respect to water availability and for
# the temperature it is more uncertain but we can keep temperature of wettest
# quarter.
reduced_datasets = []
keep_columns = ["mean_twq","mean_pet","mean_ai","cont_id"]
for i,df in enumerate(balanced_dfs):
    keep_columns.append(binned_columns[i])
    df = df[keep_columns]
    keep_columns.remove(binned_columns[i])
    reduced_datasets.append(df)

"""
10- Small arrangements before exporting: change types, remove index column.
"""

# Change cont_id from Float to Int.
for df in reduced_datasets:
    df.cont_id = df.cont_id.astype("Int64")

filename_hmean  = "./forest_data/balanced_dataset_hmean.csv"
filename_hstd   = "./forest_data/balanced_dataset_hstd.csv"
filename_crmean = "./forest_data/balanced_dataset_crmean.csv"
filename_crstd  = "./forest_data/balanced_dataset_crstd.csv"

reduced_datasets[0].to_csv(filename_hmean,index=False)
reduced_datasets[1].to_csv(filename_hstd,index=False)
reduced_datasets[2].to_csv(filename_crmean,index=False)
reduced_datasets[3].to_csv(filename_crstd,index=False)
