# -*- coding: utf-8 -*-

import glob, os, sys
import numpy as np
import pandas as pd
import geopandas as gpd

angio = gpd.read_file("/home/dibepa/Documents/TalloTreeDatabase/angiosperm_forests_tree_metric_distribution.shp")

headerTemp = ["mean_yt","mean_tcq","mean_tdq","mean_twq","std_yt","std_tcq","std_tdq","std_twq"]
for header in headerTemp:
    angio[header]  = angio[header].apply(lambda x: x + 273,15)

colMeans = ["mean_h_m","mean_cr_m","mean_yt","mean_p","mean_tcq","mean_tdq","mean_twq","mean_pwm","mean_pet","mean_ai"]
minmax = angio[colMeans].agg(["min","max"])
print(minmax)
range = (minmax.loc["max"]-minmax.loc["min"])/minmax.loc["min"]
logHeaders = range.where(lambda x: x>5.0).dropna().index.tolist()
print(logHeaders)

#back to celsius
for header in headerTemp:
    angio[header]  = angio[header].apply(lambda x: x - 273,15)

# first check logHeaders and then specify this list of columns
headerLog = ["mean_h_m","mean_cr_m","mean_p","mean_pwm","mean_pet","mean_ai","std_h_m","std_cr_m","std_p","std_pwm","std_pet","std_ai"]
for header in headerLog:
    angio[header] = angio[header].apply(lambda x: np.log(x))

# remove instances with a single sampled tree to avoid dealing with log(0) stds in the BN
# need to deal with this in a different way by merging similar points
angio = angio[angio["n_samples"]>1]

print(angio)

angioFilename = "./angiosperm_forests_tree_metric_distribution_log.shp"
angio.to_file(angioFilename, driver = "ESRI Shapefile")

gymno = gpd.read_file("/home/dibepa/Documents/TalloTreeDatabase/gymnosperm_forests_tree_metric_distribution.shp")

headerTemp = ["mean_yt","mean_tcq","mean_tdq","mean_twq","std_yt","std_tcq","std_tdq","std_twq"]
for header in headerTemp:
    gymno[header]  = gymno[header].apply(lambda x: x + 273,15)

colMeans = ["mean_h_m","std_h_m","mean_cr_m","mean_yt","mean_p","mean_tcq","mean_tdq","mean_twq","mean_pwm","mean_pet","mean_ai"]
minmax = gymno[colMeans].agg(["min","max"])
print(minmax)
range = (minmax.loc["max"]-minmax.loc["min"])/minmax.loc["min"]
logHeaders = range.where(lambda x: x>5.0).dropna().index.tolist()
print(logHeaders)

#back to celsius
for header in headerTemp:
    gymno[header]  = gymno[header].apply(lambda x: x - 273,15)

# first check logHeaders and then specify this list of columns
headerLog = ["mean_h_m","mean_cr_m","mean_p","mean_pwm","mean_pet","mean_ai","std_h_m","std_cr_m","std_p","std_pwm","std_pet","std_ai"]
for header in headerLog:
    gymno[header] = gymno[header].apply(lambda x: np.log(x))

gymno = gymno[gymno["n_samples"]>1]
print(gymno)

gymnoFilename = "./gymnosperm_forests_tree_metric_distribution_log.shp"
gymno.to_file(gymnoFilename, driver = "ESRI Shapefile")

print(angio[angio["n_samples"]==1])
