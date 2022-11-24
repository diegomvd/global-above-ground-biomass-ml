# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

path = "./forest_data/forests_tree_metric_distribution_log.shp"
forests = gpd.read_file(path)
print(forests)
drop_columns = ["index","geometry","biome_id","cont_id","div_id","std_ai","std_pet","std_yt","std_pwm","std_tcq","std_tdq","std_twq","std_p"]
forests = gpd.read_file(path).drop(columns=drop_columns)

print(forests)

"""
Figure 1 is a visualization of the correlations between predictors and the
learned qualities.
"""
# keep_columns=['mean_h_m',"std_h_m","mean_pet",'mean_ai',"biome"]
keep_columns=['mean_h_m',"std_h_m","mean_pet","mean_ai","mean_p","div"]
reduced_forests=forests[keep_columns]

# Only keep the well-sampled forests
reduced_forests = reduced_forests[reduced_forests["n_samples"]>60.0]
reduced_forests["n_samples"] = reduced_forests["n_samples"].apply(lambda x: np.log(x))
#reduced_forests["std_h_m"] = reduced_forests["std_h_m"]-reduced_forests["mean_h_m"] # re-scaling SD by mean in log space
print(reduced_forests)

sns.set_theme(style="ticks")
sns.pairplot(reduced_forests, hue = "div", corner=True,diag_kind="hist" )
plt.show()
