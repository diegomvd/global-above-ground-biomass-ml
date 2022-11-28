#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 16 12:24:45 2022

@author: Diego Bengochea
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import scipy
import rasterio
from shapely.geometry import Polygon
import pingouin as pg

# """
# Import the Tallo tree database as a GeoDataFrame.
# """
#
# tallo_path = "./tree_data/Tallo.csv"
# df = pd.read_csv(tallo_path)
# tallo = gpd.GeoDataFrame(
#     df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
# )
# tallo.crs = "EPSG:4326"
#
#
# """
# Remove the instances with no data for height and crown radius, and the outliers.
# In the future it could be possible to complete the instances with no-data using
# allometries between tree height and crown radius to avoid loosing almost 200.000
# data points.
# """
#
# tallo = tallo.drop(
#     tallo[ (tallo.height_m < 0.0) & (tallo.crown_radius_m < 0.0) ].index
# )
# tallo = tallo.drop(
#     tallo[ (tallo.height_m < 0.0) & (tallo.stem_diameter_cm < 0.0) ].index
# )
#
# tallo = tallo.drop(
#     tallo[ (tallo.height_outlier == "Y") | (tallo.crown_radius_outlier == "Y")].index
# )
#
# """
# Select the relevant columns: "height_m", "crown_radius_m", "geometry". In this
# first and general attempt of estimating global AGB stocks, differentiation by
# division, family and species is ignored.
# """
#
# keep_columns = ["stem_diameter_cm", "height_m", "crown_radius_m", "geometry"]
# tallo = tallo[keep_columns].reset_index(drop=True)
#
# print(tallo)
#
# """
# Add bio-climatic variables to the dataset by intercepting their value at each
# sampling point.
# """
#
# # Store the paths to the raster files containing the bioclomatic data.
#
# yearly_temperature           = "./bioclimatic_data/wc2.1_5m_1970_2000_annual_mean_temp.tif"
# dry_quarter_temperature      = "./bioclimatic_data/wc2.1_5m_1970_2000_mean_temp_dry_quarter.tif"
# wet_quarter_temperature      = "./bioclimatic_data/wc2.1_5m_1970_2000_mean_temp_wet_quarter.tif"
# cold_quarter_temperature     = "./bioclimatic_data/wc2.1_5m_1970_2000_mean_temp_cold_quarter.tif"
# yearly_precipitation         = "./bioclimatic_data/wc2.1_5m_1970_2000_annual_precipitation.tif"
# wet_month_precipitation      = "./bioclimatic_data/wc2.1_5m_1970_2000_precipitation_wet_month.tif"
# potential_evapotranspiration = "./bioclimatic_data/et0_v3_yr.tif"
# aridity_index                = "./bioclimatic_data/ai_v3_yr.tif"
#
# bioclimatic_data_paths = [
#     (yearly_temperature,"year_temp"),
#     (dry_quarter_temperature,"dryq_temp"),
#     (wet_quarter_temperature,"wetq_temp"),
#     (cold_quarter_temperature,"coldq_temp"),
#     (yearly_precipitation,"year_prec"),
#     (wet_month_precipitation,"wetm_prec"),
#     (potential_evapotranspiration,"pet"),
#     (aridity_index,"ai")
# ]
#
# # Extract the coordinates list from the Tallo point layer.
#
# coord_list = [(x,y) for x,y in zip(tallo['geometry'].x , tallo['geometry'].y)]
#
# for path,col in bioclimatic_data_paths:
#     raster = rasterio.open(path)
#     tallo[col] = [x for x in raster.sample(coord_list)]
#
# df = pd.DataFrame(tallo)
#
# df = df.explode(
#     ["year_temp",
#      "dryq_temp",
#      "wetq_temp",
#      "coldq_temp",
#      "year_prec",
#      "wetm_prec",
#      "pet",
#      "ai"]
# )
#
# tallo = gpd.GeoDataFrame(df).reset_index(drop=True)
# print(tallo)
# tallo = tallo.astype(
#     {"year_temp":"float",
#      "dryq_temp":"float",
#      "wetq_temp":"float",
#      "coldq_temp":"float",
#      "year_prec":"float",
#      "wetm_prec":"float",
#      'pet': 'float',
#      "ai":"float"}
# )
# print(tallo.dtypes)
# tallo.to_file("./temp_data/tallo_bioclim.shp", driver = "ESRI Shapefile")
#
# """
# Add continental region and biome to the tallo database.
# """
#
# # Load the shapefile with the continental regions and perform the spatial join.
#
# tallo["geometry"] = tallo.geometry.to_crs("EPSG:4326")
#
# continents = gpd.read_file("./continent_data/continental_regions.shp")
# continents = continents.rename({"REGION":"continent"},axis="columns")
# tallo = gpd.sjoin(tallo, continents).reset_index(drop=True)
# tallo = tallo.rename({"index_right":"cont_id"},axis="columns")
# print(tallo)
# tallo.to_file("./temp_data/tallo_bioclim_continent.shp", driver = "ESRI Shapefile")
#
# # Load the shapefile with the biomes and perform the spatial join.
#
# biomes = gpd.read_file("./biome_data/Ecoregions2017.shp")
# biomes = biomes[["BIOME_NAME","geometry"]]
# biomes = biomes.rename({"BIOME_NAME":"biome"},axis="columns")
# tallo = gpd.sjoin(tallo, biomes).reset_index(drop=True)
# tallo = tallo.rename({"index_right":"biome_id"},axis="columns")
#
# print(tallo)
# tallo.to_file("./temp_data/tallo_bioclim_continent_biome.shp", driver = "ESRI Shapefile")
#
"""
Group trees in forests.

The rationale is to create buffer areas of a determined radius around every
sampled tree and join trees whose buffers intersect. This is not done until the
very end of the process, on the meantime, each tree gets associated a site id
that is later used for grouping.

Buffer zones are only merged if the number of trees in the site is lower than a
threshold that we fix at 50 samples. Buffer radius is progressively expanded in
succeding iterations until all the sites have at least 50 samples, or the buffer
radius exceeds a defined threshold that we fix at 200km.

Using buffer zones to merge sampled trees has the limitation that at a certain
threshold the intersection might cross a percolation threshold in regions that
are densely sampled and with a low number of samples in each point. This causes
immense geographical areas merged together, reducing the pertinence of the
bioclimatic variables and the number of instances for learning.

Thus a test is implemented and if a merge results in the creation of a too big
grouping the process is aborted. In our case, this happens at the 5th iteration
with a radius of 8km. Thus only the grouping at 6km is retained.

To continue grouping we proceed to group samples by creating a grid and merging
by cell. This is not done since the beginning because it can create artificial
separations because of the arbitrary polygon borders. Again, the grid spacing is
iteratively updated until a limit of 200km. Ideally, well sampled sites are
ignored in this process. This could introduce some artifacts like trees being
merged in groups where they are not closest neighbors.

It is tested that variance in tree morphology is lower within site than within
biome. Variance in tree morphology is relatively constant over biomes. It also
is the ratio mean/median, for which its proximity to 1 suggests that the
distribution of morphologies is at least close to normal.
"""

# # Loading the Tallo database.
# tallo = gpd.read_file("./temp_data/tallo_bioclim_continent_biome.shp")
# # tallo = tallo.head(300)
#
# # Reproject Tallo in Web Mercator coordinate system to create buffer in m.
# tallo_sample = tallo.to_crs("EPSG:3857")
#
# # The first merge is done outside of the loop and its aim is to put together all
# # the trees sampled in the same point or very close. This is why the initial
# # buffer zone is set to 1km.
#
# print("Starting merging initialization...")
#
# # Create the buffer zones.
# radius = 1000 # in meters
# tallo_sample['geometry'] = tallo_sample['geometry'].buffer(radius)
#
# # Dissolve all intersecting buffers.
# tallo_sample = tallo_sample.dissolve().explode(index_parts=True).reset_index(drop=True)
#
# # Assign and id to each sampling site assimilated to each intersecting buffer.
# tallo_sample["site_id"] = range(tallo_sample.shape[0])
# tallo_sample = tallo_sample[["site_id",'geometry']]
#
# # Convert back to EPSG:4326.
# tallo_sample = tallo_sample.to_crs("EPSG:4326")
#
# # Join the buffer information in Tallo dataset.
# tallo = gpd.sjoin(tallo, tallo_sample).reset_index(drop=True)
# tallo = tallo.drop("index_right",axis="columns")
#
# # Count number of samples in each site and create a tree ID.
# tallo["n_samples"] = tallo["site_id"].map(tallo["site_id"].value_counts())
# tallo["tree_id"] = range(tallo.shape[0])
#
# print("...done.")
#
# print("Saving the temporal file.")
# # Save the intermediate shapefile.
# tallo.to_file("./temp_data2/tallo_sampling_site_r1000.shp", driver = "ESRI Shapefile")

# Now that the sites are initialized, we proceed with the iterative process to
# progressively merge more trees.
# print("Starting the merging process.")
# print()
#
# # Load the shapefile with the sites IDs initialized.
# # tallo = gpd.read_file("./temp_data2/tallo_sampling_site_r1000.shp")
#
# # Reduce the dataset to the important columns to avoid manipulating too large
# # dataframes.
# tallo = tallo[["geometry","biome","tree_id","site_id","n_samples"]]
#
# Prepare the iterative grouping of samples by increasing buffer size.
r = 2000 # base radius in meters.
dr = 1000 # the increase in buffer radius of 1km per step.
r_max = 50000 # the maximum radius for merging is 50km, in total this will
              # produce a maximum of 50 iterations.
min_sample = 50 # the minimum number of trees in a forest.
sample_threshold = 100 # merging is aborted if a site is created containing more
                      # than this number times the average number of samples in
                      # base sites.
dlmax = 100000 # maximum side of the bounding box is 100km
stop = True
# # tallo = tallo.head(3000)
# print("\n" + str(tallo.shape))
# it = 0
# # print(tallo.tree_id.value_counts(ascending=True))
# # print(tallo[tallo.tree_id == 85714.0])
# # Caution: this loop can be very long!
# # Keep looping until all the sites have a minimum of samples or the maximum
# # radius hasn't been exceeded.
# while (tallo.drop(tallo[tallo.n_samples > min_sample].index).size > 0) & (r<=r_max):
#     # Update the iteration index.
#     it = it + 1
#     print("Iteration " + str(it) + " out of maximum of 49.")
#
#     print("Re-projecting to Web Mercator.")
#     # Reproject CRS to Web Mercator to work with distances.
#     tallo_sample = tallo.to_crs("EPSG:3857")
#
#     print("Selecting lowly sampled groups (<50).")
#     # Keep only the trees that are in a group with a low number of samples.
#     tallo_sample = tallo_sample.drop(tallo_sample[tallo_sample.n_samples > min_sample].index)
#
#     print("Creating the buffer layer...")
#     # Create a new dataframe where the buffer regions of each point are merged
#     # if they intersect.
#     tallo_buffer = gpd.GeoDataFrame(tallo_sample)
#     tallo_buffer['geometry'] = tallo_sample['geometry'].buffer(r)
#     tallo_buffer = tallo_buffer[["biome","geometry"]] # keep only the important columns
#     print("...done.")
#
#     print("Start dissolving by biome...")
#     # Perform the dissove by biome to ensure that buffers from different biomes
#     # are not merged together.
#     tallo_buffer = tallo_buffer.dissolve("biome",as_index = False).explode().reset_index(drop=True)
#     print("...done.")
#
#     # Create a new column storing the id of the merged buffers.
#     tallo_buffer["buffer_id"] = range(tallo_buffer.shape[0])
#
#     print("Start the spatial join bwtween the reduced and buffered dataset...")
#     # Iterate over biomes to perform the joins only when buffer and point biome
#     # match.
#
#     biomes = tallo_sample.biome.unique()
#     # Create list of dataframes per biome to stack them later.
#     tallo_biome_list = []
#     for biome in biomes:
#         # Spatial join of the reduced Tallo dataset and the merged buffers.
#         tallo_biome_list.append(
#             gpd.sjoin(
#                 tallo_sample[tallo_sample.biome==biome],
#                 tallo_buffer[tallo_buffer.biome==biome]
#             ).reset_index(drop=True)
#         )
#     tallo_sample = pd.concat(tallo_biome_list, ignore_index=True,axis=0)
#     tallo_sample = tallo_sample.drop("index_right",axis="columns")
#     print("...done")
#
#     print("Update the sites IDs...")
#     # Get all different ids of the merged buffers.
#     sites = tallo_sample.buffer_id.unique()
#
#
#     print("Start checking if buffers percolate...")
#     # check_by_sample = False
#     # if check_by_sample == True:
#     #     # Check if too many samples are being merged together and skip the merge if
#     #     # that's the case.
#     #     for site in sites:
#     #         total_samples = tallo_sample[tallo_sample.buffer_id==site].n_samples.sum()
#     #         avg_site_samples = tallo_sample[tallo_sample.buffer_id==site].groupby(["site_id"]).n_samples.mean()
#     #         # If the threshold is exceeded the buffer_id is set to NaN and thus
#     #         # ignored for the rest of the process.
#     #         if (total_samples/avg_site_samples > sample_threshold).all() :
#     #             tallo_sample["buffer_id"] = np.where(
#     #                 (tallo_sample['buffer_id'] == site), np.nan, tallo_sample['buffer_id']
#     #             )
#     #         # If the threshold is exceeded every time, then stop is never set to
#     #         # False and the process stops, since no further merging can be done.
#     #         if (total_samples/avg_site_samples <= sample_threshold).all():
#     #             stop = False # stop is True only if the condition is always False.
#     #
#
#     check_by_bounding_box = True
#     if check_by_bounding_box == True :
#         for site in sites:
#             xmin, ymin, xmax, ymax = tallo_sample[tallo_sample.buffer_id==site].total_bounds
#             # If the side of the bounding box is too large then trees are not
#             # merged. The condition functions the other way around:
#             # if the condition is not stop is set to false and the loop continues.
#             if (not (abs(xmax-xmin)>dlmax) and not (abs(ymax-ymin)>dlmax)):
#                 stop = False
#                 continue
#             tallo_sample["buffer_id"] = np.where(
#                 (tallo_sample['buffer_id'] == site), np.nan, tallo_sample['buffer_id']
#             )
#     print("...done.")
#
#     # Exit the process of merging buffers to pass to aggregating by grid.
#     if stop == True:
#         print("No further buffers can be merged, exiting the process.")
#         break
#
#     print("Re-projecting geometries back to degrees coordinates.")
#     # Reproject the geometries of both GeoDataFrames to ESPG:4326 for the join.
#     tallo_sample = tallo_sample.to_crs("EPSG:4326")
#     tallo_buffer = tallo_buffer.to_crs("EPSG:4326")
#
#     print("Start the merging of indices...")
#     # Iterate over the merged buffers and check whether trees with identical
#     # merged buffers ids belong to different sites, if that is the case group
#     # them in the same site by maintaining the first appearing "site_id"
#     sites = tallo_sample.buffer_id.unique()
#     sites_no_nan = sites[np.isfinite(sites)] # this removes the NaN values.
#     for site in sites_no_nan:
#         # Get first appearing site id.
#         id = tallo_sample[tallo_sample.buffer_id == site].site_id.head(1)
#         # Set the same site id for all the trees sharing an identical "buffer_id".
#         tallo_sample['site_id'] = np.where(
#             (tallo_sample['buffer_id'] == site) & (tallo_sample["biome_left"]==tallo_sample["biome_right"]), id, tallo_sample['site_id']
#         )
#         # Update the number of samples in each site.
#         tallo_sample["n_samples"] = tallo_sample["site_id"].map(tallo_sample["site_id"].value_counts())
#     print("...done.")
#
#     # Now the site ids and number of samples must be updated in the entire Tallo
#     # dataset. This is done by performing a join between the reduced dataset and
#     # the entire dataset.
#
#     # In the reduced dataset, set as index the tree id.
#     tallo_sample = tallo_sample.set_index("tree_id")
#     # Only keep the columns with the updated values.
#     tallo_sample = tallo_sample[["site_id","n_samples"]]
#
#     print("Join Tallo dataset with the reduced one...")
#     tallo = tallo.join(tallo_sample, on="tree_id",rsuffix="2")
#     print("...done.")
#
#     print("Update Tallo dataset.")
#     tallo["site_id"] = np.where(tallo['site_id2'].isna(), tallo['site_id'], tallo['site_id2'])
#     tallo["n_samples"] = np.where(tallo['n_samples2'].isna(), tallo['n_samples'], tallo['n_samples2'])
#     tallo = tallo.drop(["site_id2","n_samples2"],axis="columns")
#
#     print("Number of forests " + str(tallo.site_id.unique().size) + ".")
#     tallo = tallo.rename({"biome_left":"biome"},axis="columns")
#
#     print("Summary info:")
#     print(tallo.groupby(["site_id"]).size())
#     print(tallo.groupby(["site_id"]).size().min())
#     print(tallo.groupby(["site_id"]).size().max())
#     print(tallo.groupby(["site_id"]).size().mean())
#     print(tallo.groupby(["site_id"]).size().median())
#
#     print(tallo.shape)
#
#     print("Save updated layer.")
#     tallo.to_file("./temp_data_grouping_buffer/tallo_sampling_site_r"+ str(r) +".shp", driver = "ESRI Shapefile")
#
#     print("Expand radius.")
#     r = r + dr

# tallo = gpd.read_file("./temp_data_grouping_buffer/tallo_sampling_site_r29000.shp")
#
# print("Number of forests " + str(tallo.site_id.unique().size) + ".")
# tallo = tallo.rename({"biome_left":"biome"},axis="columns")
# print("Summary info:")
# print(tallo.groupby(["site_id"]).size())
# print(tallo.groupby(["site_id"]).size().min())
# print(tallo.groupby(["site_id"]).size().max())
# print(tallo.groupby(["site_id"]).size().mean())
# print(tallo.groupby(["site_id"]).size().median())
# print(tallo.shape)
#
# # Continue the process by aggregating in a grid.
# print("Start merging by grid.")
# # tallo = gpd.read_file("./temp_data2/tallo_sampling_site_r1000.shp")
# r = 29000
# l = r # start with a tile side-length equal to the last tried buffer radius.
# dl = 1000 # use an increment of 1km.
# l_max = 100000 # the maximum cell side for merging is 100km, in total this will
#                # produce a maximum of 200 iterations.
# it = 0
#
# # Caution: this loop can be very long!
# # Keep looping until all the sites have a minimum of samples or the maximum
# # radius hasn't been exceeded.
# print()
# while (tallo.drop(tallo[tallo.n_samples > min_sample].index).size > 0) & (l<l_max):
#
#     # Update the iteration index.
#     it = it + 1
#     print("Iteration " + str(it) + " out of maximum of 100.")
#
#     print("Re-projecting to Web Mercator.")
#     # Reproject CRS to Web Mercator to work with distances.
#     tallo_sample = tallo.to_crs("EPSG:3857")
#
#     print("Selecting lowly sampled groups (<50).")
#     # Keep only the trees that are in a group with a low number of samples.
#     tallo_sample = tallo_sample.drop(tallo_sample[tallo_sample.n_samples > min_sample].index)
#
#     print("Start building the grid.")
#     # Create a grid with the bounds of the reduced dataset.
#     # Adapted from:
#     # https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
#     xmin, ymin, xmax, ymax = tallo_sample.total_bounds
#
#     cols = list(np.arange(xmin, xmax + l, l))
#     rows = list(np.arange(ymin, ymax + l, l))
#
#     print("Create the tiles.")
#     polygons = []
#     for x in cols[:-1]:
#         for y in rows[:-1]:
#             polygons.append(Polygon([(x,y), (x+l, y), (x+l, y+l), (x, y+l)]))
#
#     print("Create the GeoDataFrame with the grid.")
#     grid = gpd.GeoDataFrame({'geometry':polygons}).set_crs("EPSG:3857")
#
#     print("Associate indices with the polygons.")
#     grid["tile_id"] = range(grid.shape[0])
#
#     print("Re-projecting geometries back to degrees coordinates.")
#     # Reproject the geometries of both GeoDataFrames to ESPG:4326 for the join.
#     tallo_sample = tallo_sample.to_crs("EPSG:4326")
#     grid = grid.to_crs("EPSG:4326")
#
#     print("Start the spatial join.")
#     tallo_sample = gpd.sjoin(tallo_sample, grid).reset_index(drop=True)
#
#     print("Update the indices by iterating over tiles and biomes...")
#     tiles = tallo_sample.tile_id.unique()
#     biomes = tallo_sample.biome.unique()
#     # print(biomes)
#     # print(tiles)
#     # sites_no_nan = sites[np.isfinite(sites)]
#     for tile in tiles:
#         for biome in biomes: # This is to separate biomes in the same tile.
#             # Get first appearing site id.
#             id = tallo_sample[(tallo_sample.tile_id == tile) & (tallo_sample.biome == biome)].site_id.head(1)
#             if id.empty:
#                 continue
#             # Set the same site id for all the trees sharing an identical "buffer_id".
#             tallo_sample['site_id'] = np.where(
#                 (tallo_sample['tile_id'] == tile), id, tallo_sample['site_id']
#             )
#             # Update the number of samples in each site.
#             tallo_sample["n_samples"] = tallo_sample["site_id"].map(tallo_sample["site_id"].value_counts())
#     print("...done.")
#
#     # Now the site ids and number of samples must be updated in the entire Tallo
#     # dataset. This is done by performing a join between the reduced dataset and
#     # the entire dataset.
#
#     # In the reduced dataset, set as index the tree id.
#     tallo_sample = tallo_sample.set_index("tree_id")
#     # Only keep the columns with the updated values.
#     tallo_sample = tallo_sample[["site_id","n_samples"]]
#
#     print("Join Tallo dataset with the reduced one...")
#     tallo = tallo.join(tallo_sample, on="tree_id",rsuffix="2")
#     print("...done.")
#
#     print("Update Tallo dataset.")
#     tallo["site_id"] = np.where(tallo['site_id2'].isna(), tallo['site_id'], tallo['site_id2'])
#     tallo["n_samples"] = np.where(tallo['n_samples2'].isna(), tallo['n_samples'], tallo['n_samples2'])
#     tallo = tallo.drop(["site_id2","n_samples2"],axis="columns")
#
#     print("Number of forests " + str(tallo.site_id.unique().size) + ".")
#
#     print("Summary info:")
#     print(tallo.groupby(["site_id"]).size())
#     print(tallo.groupby(["site_id"]).size().min())
#     print(tallo.groupby(["site_id"]).size().max())
#     print(tallo.groupby(["site_id"]).size().mean())
#     print(tallo.groupby(["site_id"]).size().median())
#
#     print("Save updated layer.")
#     tallo.to_file("./temp_data_grouping_grid/tallo_sampling_site_r"+ str(r) +"_l"+str(l)+".shp", driver = "ESRI Shapefile")
#     print(tallo.shape)
#
#     print("Increase tile size.")
#     l = l + dl

"""
Join again the data on merged sites with the bioclimatic data
"""

tallo_full = gpd.read_file("./temp_data_grouping_buffer/tallo_sampling_site_r1000.shp")
tallo_sites = gpd.read_file("./temp_data_grouping_grid/tallo_sampling_site_r29000_l99000.shp")

# From tallo_site keep only tree_id, site_id and n_samples columns. tree_id is
# for merging on that index and site_id and n_samples to update the information
# in tallo_full.

tallo_sites = tallo_sites[["tree_id","site_id","n_samples"]]
tallo_full = tallo_full.drop(["site_id","n_samples"],axis="columns")
tallo = tallo_full.merge(tallo_sites, on="tree_id")

# Replace no_data values by np.nan
no_data = -9999.9999
tallo = tallo.replace(-9999.9999, np.nan)

print(tallo[tallo.stem_diame.isna()].shape[0])
print(tallo[tallo.crown_radi.isna()].shape[0])
print(tallo[tallo.height_m.isna()].shape[0])

print( tallo[ (tallo.crown_radi.isna()) & (tallo.stem_diame.isna()) ].shape[0])
print( tallo[ (tallo.height_m.isna()) & (tallo.stem_diame.isna()) ].shape[0])
print( tallo[ (tallo.height_m.isna()) & (tallo.crown_radi.isna()) ].shape[0])

"""
Fill crown-radius data with height and diameter allometry based from:
Jucker et al. 2016, Allometric equations for integrating remote sensing imagery
into forest monitoring programmes. Global Change Biology.
"""

def crown_radius(height, diameter):
    return np.exp(-0.5/0.809*0.056**2) / np.power(0.557,1/0.809) * np.power(diameter,1/0.809) / height

def height(crown_radius, diameter):
    return np.exp(-0.5/0.809*0.056**2) / np.power(0.557,1/0.809) * np.power(diameter,1/0.809) / crown_radius

tallo["crown_radi"] = np.where(
    tallo['crown_radi'].isna(),
    crown_radius(tallo["height_m"],tallo["stem_diame"]),
    tallo['crown_radi']
)
tallo["height_m"] = np.where(
    tallo['height_m'].isna(),
    crown_radius(tallo["crown_radi"],tallo["stem_diame"]),
    tallo['height_m']
)

# """
# Estimate the empirical bivariate distribution of trees' morphology in each site
# with enough samples. The process involves the discretization of the height-crown
# radius domain for a range of cell sizes. The resulting distribution is fitted
# by GAM or Gaussian Process Regression and then the values predicted by the fitted
# distribution are compared with the original samples. Based on the accuracy
# obtained, a discretization of the height-crown radius domain is kept for each
# site and the best regression is kept to estimate the AGB/ha in the site.
# """
#
# for site in tallo.site_id.unique():
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract tree height and crown radius data.
#     X = tallo_site[["height_m","crown_radi"]]
#     # Initialize de grid resolution, resolution step and minimum resolution.
#     lmin  = 0.1
#     dl = 0.1
#     lmax = 10.0
#     lvec = np.arange(lmin,lmax,dl)
#     # Initialize minimums and maximums for the H-CR domain.
#     hmin = 0.0
#     hmax = 10.0
#     crmin = 0.0
#     crmax = 10.0
#     # Start the iterations to progressively decrement grid resolution.
#     for l in lvec:
#         hvec  = np.arange(hmin,hmax,l)
#         crvec = np.arange(crmin,crmax,l)
#         # Initialize the distribution matrix
#         distribution = []
#         # Start the iterations to count sample occurrence in the discretized
#         # domain.
#         for h in hvec:
#             for cr in crvec:
#                 matchs = X.where( (X.height_m > h-l) & (X.height_m < h+l) &
#                             (X.crown_radi > cr-l) & (X.crown_radi < cr+l),
#                             1.0, 0.0)
#                 z = X.height_m.sum(axis=1)
#                 distribution.append([h,cr,z])
#         print(distribution)
#         distribution = np.array(distribution)
#
#         # Create a Gaussian Process Regressor to fit the the obtained empirical
#         # distribution.
#         fit =
#         domain = # This is the domain to produce points by evaluating the regressor.
#         f = # This is the evaluated function.
#
#         # Evaluate the fit function on the original data points: this is done by
#         # integrating the fitted function over multiple random domains and
#         # comparing the result with the real number of occurrences in such domain.

"""
Estimate the empirical bivariate distribution of trees' morphology in each site
with enough samples. The process involves a kernel-density estimation of the
height-crown radius distribution. The outcome od the process is a dictionary
from site_id to best parameters for the KDE.
"""

site_param_dict = {}
for site in tallo.site_id.unique():
    tallo_site = tallo[tallo.site_id == site]
    # Extract tree height and crown radius data.
    X = tallo_site[["height_m","crown_radi"]].to_numpy()
    # Initialize the bandwidth vector to test.
    bw_min = 0.3
    bw_max = 2.0
    bw_d   = 0.1
    bw_vec = np.arange(bw_min,bw_max,bw_d)
    # Initialize the maximum log_likelihood
    ll_max = -999999.0
    # Start iterations for testing KDE parametrizations: better use GridSearchCV
    for bw in bw_vec:
        # Build the KDE instance.
        kde = KernelDensity(kernel = 'gaussian', bandwidth = bw).fit(X)
        log_likelihood = kde.score_samples(X)
        if log_likelihood > ll_max :
            ll_max = log_likelihood
            params = (k,bw)
    site_param_dict[site] = params

"""
Join tree density data and calculate average tree density in every site to start
generating forests based on the estimated distributions of height and crown
radius. This requires simulating the N trees and calculating total biomass M
times to then average. The std of the observation can be kept as information on
the possible spread.
"""

# Add tree density information to the dataset.

tree_density = "./tree_density_data/tree_density_data/tree_density_biome_based_model_crowther_nature_2015_4326_float32.tiff"

# Extract the coordinates list from the Tallo point layer.
coord_list = [(x,y) for x,y in zip(tallo['geometry'].x , tallo['geometry'].y)]

raster = rasterio.open(tree_density)
tallo["tree_dens"] = [x for x in raster.sample(coord_list)]

df = pd.DataFrame(tallo)
df = df.explode( "tree_dens" )
tallo = gpd.GeoDataFrame(df).reset_index(drop=True)
tallo = tallo.astype( {"tree_dens":"float"} )
print(tallo)
# tallo.to_file("./temp_data/tallo_bioclim.shp", driver = "ESRI Shapefile")

# Simulate tree morphologies' occurrences based on the KDEs and calculate
# average AGB/ha in each site.
agb_dict = {}
for site in tallo.site_id.unique():
    tallo_site = tallo[tallo.site_id == site]
    # Extract tree height and crown radius data.
    X = tallo_site[["height_m","crown_radi"]].to_numpy()
    # Calculate the average number of samples needed to simulate AGB in site.
    n_samples = tallo_site.tree_dens.mean()
    # Esimate the distribution with the optimum parameters.
    kde = KernelDensity( kernel = 'gaussian',
                         bandwidth = site_param_dict[site] ).fit(X)
    # Initialize AGB list.
    agb_list = []
    replication = 100
    for i in range(replication):
        # Initialize AGB
        agb = 0
        # Generate the tree instances
        trees = kde.sample(n_samples)
        for tree in trees:
            h = tree[0]
            cr = tree[1]
            agb += (above_ground_biomass(h,cr)
        agb_list.append(agb)
    median_agb = np.median(np.array(agb))
    agb_dict[site] = median_agb

"""
Dissolve the data per site and add the AGB
"""









# print(tallo[ tallo.crown_radi.isna() ].shape)
#
# """
# Test for bivariate log-normality in each site
# """
#
# print("Testing joint log-normality with HZ test in the grouped dataset.")
# positive = 0
# negative = 0
# for site in tallo.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site[["height_m","crown_radi"]].to_numpy()
#     if X.shape[0] >= 20:
#         X = np.log(X)
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         test_result = pg.multivariate_normality(X,alpha=.001)
#         if test_result[2] == True:
#             positive += 1
#         else:
#             negative += 1
#     # print(test_result)
# print(positive)
# print(negative)
# print(positive/negative)
#
# """
# Test for bivariate normality in each site
# """
#
# print("Testing joint normality with HZ test in the grouped dataset.")
# positive = 0
# negative = 0
# for site in tallo.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site[["height_m","crown_radi"]].to_numpy()
#     if X.shape[0] >= 20:
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         test_result = pg.multivariate_normality(X,alpha=.001)
#         if test_result[2] == True:
#             positive += 1
#         else:
#             negative += 1
#     # print(test_result)
# print(positive)
# print(negative)
# print(positive/negative)


# print("Testing joint normality with HZ test in the initial dataset.")
# tallo_init = gpd.read_file("./temp_data_grouping_buffer/tallo_sampling_site_r1000.shp")
#
# positive = 0
# negative = 0
# for site in tallo_init.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo_init[tallo_init.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site[["height_m","crown_radi"]].to_numpy()
#     if X.shape[0] >= 50:
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         test_result = pg.multivariate_normality(X,alpha=.05)
#         if test_result[2] == True:
#             positive += 1
#         else:
#             negative += 1
#     # print(test_result)
# print(positive)
# print(negative)
# print(positive/negative)


"""
Test for univariate normality of tree height and crown radius independently
"""

# print("Testing normality of tree height: d'agostino")
# positive = 0
# negative = 0
# for site in tallo.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site["height_m"].to_numpy()
#     if X.shape[0] >= 20:
#         X = np.log(X)
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         k2,p = scipy.stats.normaltest(X)
#         if p >= 0.05 == True:
#             positive += 1
#         else:
#             negative += 1
# print(positive/negative)
#
# print("Testing normality of crown radius: d'agostino")
# positive = 0
# negative = 0
# for site in tallo.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site["crown_radi"].to_numpy()
#     if X.shape[0] >= 20:
#         X = np.log(X)
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         k2,p = scipy.stats.normaltest(X)
#         if p >= 0.05 == True:
#             positive += 1
#         else:
#             negative += 1
# print(positive/negative)
#
#
#
# print("Testing normality of tree height: measure skewness and kurtosis")
# positive = 0
# negative = 0
# for site in tallo.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site["height_m"].to_numpy()
#     if X.shape[0] >= 10:
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         mean = np.mean(X)
#         median = np.median(X)
#         if (mean/median>0.75 and mean/median<1.25) == True:
#             positive += 1
#         else:
#             negative += 1
# print(positive)
# print(negative)
#
# print("Testing normality of crown_radi: measure skewness and kurtosis")
# positive = 0
# negative = 0
# for site in tallo.site_id.unique():
#     # print("Site : " + str(site))
#     tallo_site = tallo[tallo.site_id == site]
#     # Extract the data to test: tree height and crown radius
#     X = tallo_site["crown_radi"].to_numpy()
#     if X.shape[0] >= 10:
#         # print("Number of samples: " + str(X.shape[0]) + ", biome: " + str(tallo_site.head(1).biome))
#         mean = np.mean(X)
#         median = np.median(X)
#         if (mean/median>0.75 and mean/median<1.25) == True:
#             positive += 1
#         else:
#             negative += 1
# print(positive)
# print(negative)

# """
# Merge the data: calculate mean, sd and convariance of height and crown radius.
# """
# print(tallo.columns)
#
# # First calculate the mean and sd.
#
# tallo_forest = tallo.dissolve(
#     by = "site_id",
#     aggfunc = {
#         "stem_diame": ["mean",np.std],
#         "height_m": ["mean",np.std],
#         "crown_radi": ["mean",np.std],
#         "year_temp": ["mean",np.std],
#         "dryq_temp": ["mean",np.std],
#         "wetq_temp": ["mean",np.std],
#         "coldq_temp": ["mean",np.std],
#         "year_prec": ["mean",np.std],
#         "wetm_prec": ["mean",np.std],
#         "pet": ["mean",np.std],
#         "ai": ["mean",np.std],
#         "cont_id": "first",
#         "continent": "first",
#         "biome_id": "first",
#         "biome": "first",
#         "tree_id" : "first",
#         "n_samples" : "first"
#     }
# )

# Now calculate the covariance between H and CR and merge with tallo_forest

# tallo = gpd.read_file("./temp_data_grouping_buffer/tallo_sampling_site_r29000.shp")


# """
# Testing produced shapefiles
# """
#
# tallo = gpd.read_file("./temp_data/tallo_sampling_site_iteration_3.shp")
#
# # print()
# # print("Final number of forests:")
# # print(tallo.site_id.unique().size)
# #
# # # Summary statistics
# # print(tallo.groupby(["site_id"]).size())
# # print(tallo.groupby(["site_id"]).size().min())
# # print(tallo.groupby(["site_id"]).size().max())
# # print(tallo.groupby(["site_id"]).size().mean())
# # print(tallo.groupby(["site_id"]).size().median())
# #
# # # By biome
# # for biome in tallo.biome.unique():
# #     # tot = tallo[tallo.biome==biome].groupby(["site_id"]).size()
# #     min = tallo[tallo.biome==biome].groupby(["site_id"]).size().min()
# #     max = tallo[tallo.biome==biome].groupby(["site_id"]).size().max()
# #     mean = tallo[tallo.biome==biome].groupby(["site_id"]).size().mean()
# #     median = tallo[tallo.biome==biome].groupby(["site_id"]).size().median()
# #     print("Biome: " + str(biome))
# #     # print("Total = " +str(tot))
# #     print("Min = " +str(min))
# #     print("Max = " +str(max))
# #     print("Mean = " +str(mean))
# #     print("Median = " +str(median))
# tallo = tallo.replace(-9999.9999, np.nan)
#
# # print("Morphologies' standard deviations in:")
# # for biome in tallo.biome.unique():
# #     # tot = tallo[tallo.biome==biome].groupby(["site_id"]).size()
# #     mh = tallo[tallo.biome==biome].height_m.mean()
# #     mc = tallo[tallo.biome==biome].crown_radi.mean()
# #     sdh = tallo[tallo.biome==biome].height_m.std()
# #     sdc = tallo[tallo.biome==biome].crown_radi.std()
# #     print("Biome: " + str(biome))
# #     # print("Total = " +str(tot))
# #     print("height: " +str(sdh/mh))
# #     print("Crown radius : " +str(sdc/mc))
# #     print()
# #
# # print("Morphologies' mean/median")
# # for biome in tallo.biome.unique():
# #     # tot = tallo[tallo.biome==biome].groupby(["site_id"]).size()
# #     mh = tallo[tallo.biome==biome].height_m.mean()
# #     mc = tallo[tallo.biome==biome].crown_radi.mean()
# #     medh = tallo[tallo.biome==biome].height_m.median()
# #     medc = tallo[tallo.biome==biome].crown_radi.median()
# #     print("Biome: " + str(biome))
# #     # print("Total = " +str(tot))
# #     print("height: " +str(mh/medh))
# #     print("Crown radius : " +str(mc/medc))
# #     print()
#
# print("Morphologies' standard deviations average per site")
# for biome in tallo.biome.unique():
#     # tot = tallo[tallo.biome==biome].groupby(["site_id"]).size()
#     mh = tallo[tallo.biome==biome].groupby(["site_id"]).height_m.mean()
#     mc = tallo[tallo.biome==biome].groupby(["site_id"]).crown_radi.mean()
#     sdh = tallo[tallo.biome==biome].groupby(["site_id"]).height_m.std()
#     sdc = tallo[tallo.biome==biome].groupby(["site_id"]).crown_radi.std()
#
#     average_ratio_height = (sdh/mh).mean()
#     average_ratio_crownr = (sdc/mc).mean()
#     print("Biome: " + str(biome))
#     # print("Total = " +str(tot))
#     print("Height: " +str(average_ratio_height))
#     print("Crown radius : " +str(average_ratio_crownr))
#     print()

# print(tallo.groupby(["biome","site_id"]).n_samples.mean())
# print(tallo.groupby(["biome","site_id"]).n_samples.median())
# print(tallo.groupby(["biome","site_id"]).n_samples.min())
# print(tallo.groupby(["biome","site_id"]).n_samples.max())
