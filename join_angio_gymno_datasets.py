# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script joins forest datasets of angiosperms and gymnosperms to increase
data samples for the learning process.
It is not clear yet if for the sake of AGB prediction it is better to have worse
forest metric predictions but differentiated allometries for angiosperms and
gymnosperms, or if it s better to have a poorer allometrie for both and
increased prediction capabilities of forest metrics.

04/11/2022
dibepa
"""

file_angiosperm = "./forest_data/angiosperm_forests_tree_metric_distribution_log.shp"
file_gymnosperm = "./forest_data/gymnosperm_forests_tree_metric_distribution_log.shp"

angiosperm_forests = gpd.read_file(file_angiosperm)
gymnosperm_forests = gpd.read_file(file_gymnosperm)

# create division column
angiosperm_forests["div"] = "angiosperm"
gymnosperm_forests["div"] = "gymnosperm"

# create division_id column
angiosperm_forests["div_id"] = 1
gymnosperm_forests["div_id"] = 2

# concatenate both dataframes
forests = pd.concat([angiosperm_forests, gymnosperm_forests]).reset_index()

# keep only the forests with more than 60 trees sampled
forests = forests[forests["n_samples"]>60.0]
print(forests)

filepath = "./forest_data/forests_tree_metric_distribution_log_samples60.shp"
forests.to_file(filepath, driver = "ESRI Shapefile")
