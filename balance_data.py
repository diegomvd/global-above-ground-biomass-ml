# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geopandas as gpd
from imblearn.over_sampling import SMOTE

path = "./forest_data/forests_tree_metric_distribution_log.shp"
forests = gpd.read_file(path)

# Keep only isntances with more than 60.0 sampled trees.
forests = forests[forests["n_samples"]>10.0]

# Bin the target features
n_bins = 8

targets      = ["mean_h_m","std_h_m","mean_cr_m","std_cr_m"]
new_col_name = ["hmn_bin","hsd_bin","crmn_bin","crsd_bin"]

for i,col in enumerate(targets):
    min    = forests[col].agg(["min"])[0]
    max    = forests[col].agg(["max"])[0]
    bins   = np.linspace(min,max,n_bins).tolist()
    print(bins)
    labels = np.arange(n_bins).tolist()
    print(labels)

    new_col = new_col_name[i]
    forests[new_col] = pd.cut(forests[col],bins=n_bins,labels=labels)
    print(forests)

print(forests)

# Balance the data for each of the target features. This results in 4 different
# datasets!

# Since we have a relatively small number of samples, let's over-sample!
over_sampler = SMOTE(k_neighbors=2)

# Specify the predictors to be used.
predictors   = ["mean_yt","mean_p","mean_tcq","mean_tdq","mean_twq","mean_pwm","mean_ai","cont_id"]
predictor_df = forests[predictors]

for col in new_col_name:
    target = forests[col]
    print(target)
    predictors_balanced, target_balanced = over_sampler.fit_resample(predictor_df, target)
    print(predictors_balanced)
