# -*- coding: utf-8 -*-

import glob, os, sys
import numpy as np
import pandas as pd
import geopandas as gpd

"""
This script is used to re-scale standard deviations in the dataset with respect
to the mean.
"""

angiosperm_data = "/home/dibepa/Documents/TalloTreeDatabase/angiosperm_forests_tree_metric_distribution.shp"
gymnosperm_data = "/home/dibepa/Documents/TalloTreeDatabase/gymnosperm_forests_tree_metric_distribution.shp"

angio = gpd.read_file(angiosperm_data)
gymno = gpd.read_file(gymnosperm_data)

pair_columns  = [("mean_h_m","std_h_m"),("mean_cr_m","std_cr_m"),("mean_yt","std_yt"),("mean_p","std_p"),("mean_tcq","std_tcq"),("mean_tdq","std_tdq"),("mean_twq","std_twq"),("mean_pwm","std_pwm"),("mean_pet","std_pet"),("mean_ai","std_ai")]

for pair in pair_columns:
    mean = pair[0]
    sd   = pair[1]
    angio[sd] = angio[sd]/angio[mean]
    gymno[sd] = gymno[sd]/gymno[mean]

angioFilename = "./forest_data/angiosperm_scaledsd.shp"
gymnoFilename = "./forest_data/gymnosperm_scaledsd.shp"

print(angio)
print(gymno)

angio.to_file(angioFilename, driver = "ESRI Shapefile")
gymno.to_file(gymnoFilename, driver = "ESRI Shapefile")
