import rasterio
from rasterio.enums import Resampling
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import geopandas as gpd

# """
# Train RF with all the instances and the tuned hyper-parameters. 
# """

# # Load and prepare the final training-dataset: density, pet and agb will be in log-scale.

# # Load data: 
# data = pd.read_csv("./forest_data/forest_depurated.csv")

# # Drop unnecessary columns, considering that the 4 best features will be kept: 
# # Potential Evapo-transpiration, Tree density, Temperature of the coldest quarter, Biogeographical realm.
# data = data.drop(["Unnamed: 0", "cont_id","continent","biome","n_samples","site_id", "ai", "wmp_mm", "yp_mm", "tdq_c", "yt_c","twq_c"],axis="columns")

# # Removing too large AGBs that seem outliers originated by errors (most in Australasia).
# data = data[data.agb_tha < 10000.0 ].reset_index(drop=True)


# # Drop Oceania biogeographical realm as there is a single sample.
# data = data.drop(data[data.bioge_realm=="Oceania"].index).reset_index(drop=True)
# # print(data.bioge_realm.value_counts())

# # Perform a binary encoding for biogeographical realms: this adds 3 more columns.
# # Palearctic  = 0 0 0
# # Indomalayan = 0 1 0
# # Australasia = 0 0 1
# # Nearctic    = 0 1 1
# # Afrotropic  = 1 0 0
# # Neotropic   = 1 1 0

# data["bgr1"] = np.nan
# data["bgr2"] = np.nan
# data["bgr3"] = np.nan

# data["bgr1"] = np.where(
#                 (data["bioge_realm"] == "Afrotropic") | (data["bioge_realm"] == "Neotropic"),
#                  1, 0 )
# data["bgr2"] = np.where(
#                 (data["bioge_realm"] == "Indomalayan") | (data["bioge_realm"] == "Nearctic") | (data["bioge_realm"] == "Neotropic"),
#                  1, 0 )
# data["bgr3"] = np.where(
#                 (data["bioge_realm"] == "Nearctic") | (data["bioge_realm"] == "Australasia"),
#                  1, 0 )

# # Drop the bioge_realm column.
# data = data.drop("bioge_realm",axis="columns")

# # Log-transform variables spanning several orders of magnitude.
# data["agb_tha"] = np.log(data["agb_tha"])
# data["pet"] = np.log(data["pet"])
# data["density_km2"] = np.log(data["density_km2"])

# # Rename the log-transformed variables to keep trace of the log operation.
# data = data.rename({"agb_tha":"agb_tha_log",
#                     "pet":"pet_log",
#                     "density_km2":"density_km2_log"}, axis = "columns")

# # Order the columns to be consistent with predictors' order.
# data = data[["agb_tha_log","tcq_c","pet_log","density_km2_log","bgr1","bgr2","bgr3"]]

# # Extract the target and predictors as numpy arrays to train a random forest regressor. 
# agb = data["agb_tha_log"].to_numpy()
# bioclim_predictors = data.drop(["agb_tha_log"],axis="columns").to_numpy()

# print(agb)
# print(bioclim_predictors)

# # Hyper-parameters for the Random Forest Regressor: selected after multi-optimization with genetic algorithms. 
# n_estimators = 200
# max_depth = None
# min_samples_leaf = 1
# min_impurity_decrease = 0.0001

# # Initialize the RF.
# rf = RandomForestRegressor(n_estimators = n_estimators,
#                            max_depth = max_depth,
#                            min_samples_leaf = min_samples_leaf,
#                            min_impurity_decrease = min_impurity_decrease)

# # Train the RF with the prepared dataset.
# rf.fit(bioclim_predictors,agb)

# """
# Load predictors data and transform: logarithmize PET and TD and encode BGR. 
# """

# predictors = gpd.read_file("./predictor_global_data2.shp")
# predictors["td"] = np.log(predictors["td"])
# predictors = predictors.drop(
#     predictors[ (predictors.pet <= 0.0) ].index
# ).reset_index(drop=True)
# predictors["pet"] = np.log(predictors["pet"])
# # Rename the log-transformed variables to keep trace of the log operation.
# predictors = predictors.rename({ "pet":"pet_log",
#                                  "td":"td_log"}, axis = "columns")


# # Drop Oceania biogeographical realm as there is a single sample.
# predictors = predictors.drop(predictors[predictors.bgr=="Oceania"].index).reset_index(drop=True)
# predictors = predictors.drop(predictors[predictors.bgr=="Antarctica"].index).reset_index(drop=True)

# # Perform a binary encoding for biogeographical realms: this adds 3 more columns.
# # Palearctic  = 0 0 0
# # Indomalayan = 0 1 0
# # Australasia = 0 0 1
# # Nearctic    = 0 1 1
# # Afrotropic  = 1 0 0
# # Neotropic   = 1 1 0

# predictors["bgr1"] = np.nan
# predictors["bgr2"] = np.nan
# predictors["bgr3"] = np.nan

# predictors["bgr1"] = np.where(
#                 (predictors["bgr"] == "Afrotropic") | (predictors["bgr"] == "Neotropic"),
#                  1, 0 )
# predictors["bgr2"] = np.where(
#                 (predictors["bgr"] == "Indomalayan") | (predictors["bgr"] == "Nearctic") | (predictors["bgr"] == "Neotropic"),
#                  1, 0 )
# predictors["bgr3"] = np.where(
#                 (predictors["bgr"] == "Nearctic") | (predictors["bgr"] == "Australasia"),
#                  1, 0 )

# print(predictors)

# print(np.min(predictors["pet_log"]))
# print(np.min(predictors["td_log"]))

# # Extract the columns that are going to be predictors in the correct order.
# predictors_array = predictors[["tcq","pet_log","td_log","bgr1","bgr2","bgr3"]].to_numpy()

# """
# Predict AGB using the world's rasters.
# """

# agb_log_predicted = rf.predict(predictors_array)

# predictors["agblog_p"] = agb_log_predicted
# predictors["agb_p"] = np.exp(predictors["agblog_p"])

# print(predictors)
# print(np.unique(predictors["agb_p"]))

# # predictors.to_csv("predicted_agb.csv")
# predictors.to_file("predicted_agb.shp", driver = "ESRI Shapefile")

# Create raster-layer
# First create an array of the needed shape resolution at 5-arc minutes

dl = 0.08333
lat = np.linspace(-90,90,2160)
long = np.linspace(-180,180,4320)
X , Y = np.meshgrid(long,lat)
print(X)
print(X.shape)
print(Y.shape)

# Iterate over the predicted points and find matching coordinates, then assign value in the corresponding cell of a 2D array

agb_array = np.ones((2160,4320))*(-9999.0)

agb_predicted = gpd.read_file("predicted_agb.shp")
print(agb_predicted)
print(agb_predicted["geometry"].y)

data = np.transpose(np.array([agb_predicted["agblog_p"].to_numpy(),agb_predicted["geometry"].x.to_numpy(),agb_predicted["geometry"].y.to_numpy()]))
print(data)
print(data.shape)

for row in data: 
    agb = row[0]
    longitude = row[1]
    latitude = row[2]

    # print(agb)
    # print(longitude)
    # print(latitude)

    i = np.where((latitude + dl/3 > lat ) & (latitude-dl/3 < lat))
    j = np.where((longitude+dl/3 > long) & (longitude-dl/3 < long))

    agb_array[i,j] = agb

print(agb_array)

with rasterio.Env():

    # Write an array as a raster band to a new 8-bit file. For
    # the new file's profile, we start with the profile of the source
    profile = rasterio.open("./bioclimatic_data/wc2.1_5m_1970_2000_mean_temp_cold_quarter.tif").profile

    # And then change the band count to 1, set the
    # dtype to float32, and specify LZW compression.
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    with rasterio.open('agb_predicted.tif', 'w', **profile) as dst:
        dst.write(agb_array.astype(rasterio.float32), 1)


# for row in agb_predicted:
    
#     agb = row.agblog_p
#     longitude = row.geometry.x
#     latitude = row.geometry.y

#     print(agb)
#     print(longitude)
#     print(latitude)

#     i = np.where((latitude + dl/3 > lat ) & (latitude-dl/3 < lat))
#     j = np.where((longitude+dl/3 > long) & (longitude-dl/3 < long))

#     agb_array[i,j] = agb

# agb_predicted["i"] = np.where((agb_predicted["geometry"].y + dl/3 > lat ) & (agb_predicted["geometry"].y-dl/3 < lat))
# print(agb_predicted)
# agb_predicted["j"] = np.where((agb_predicted["geometry"].x + dl/3 > long ) & (agb_predicted["geometry"].x-dl/3 < long))
# print(agb_predicted)