"""
Extract data points for prediction from rasters: PET, TCQ, TD and polygon layer BGR:

The procedure is to get coordinates of tiles of the less resoluted raster layer then 
sample in these coordinates the more resoluted ones, and finally perform a spatial 
join with the polygon layer. 
"""

import rasterio
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import geopandas as gpd

# with rasterio.open("./tree_density_data/tree_density_biome_based_model_crowther_nature_2015_4326_float32.tiff") as dataset:

#     print(dataset.crs)

# Load raster with lower resolution (tcq) as a point layer.
with rasterio.open("./bioclimatic_data/wc2.1_5m_1970_2000_mean_temp_cold_quarter.tif") as dataset:

    print(dataset.crs)
    print(dataset.transform * (0,0))
    print(dataset.transform * (dataset.width,dataset.height))


    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * 1.0),
            int(dataset.width * 1.0)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )

    # Extract raster data and coordinates in a numpy array.
    data = data[0,:,:]
    predictor = []
    for row,vec in enumerate(data):
        for col,elem in enumerate(vec):
            xy = transform * (col,row)
            predictor.append([elem, xy[0], xy[1]])
    predictor = np.array(predictor)
    print(predictor.shape)

# Create a DataFrame from the array.
df = pd.DataFrame(predictor, columns=["tcq","x","y"])
print(df)

df = df.drop(
    df[ (df.tcq < -280.0) ].index
).reset_index(drop=True)
print(df)

# Create a GeoDataFrame using the extracted coordinates. 
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x, df.y)
).set_crs("EPSG:4326")

print(gdf)

# Get coordinates of each point.
coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]

print(coord_list)
print(len(coord_list))

# Open rasters and sample in the coordinates.
pet_path = "./bioclimatic_data/et0_v3_yr.tif"
td_path  = "./tree_density_data/tree_density_biome_based_model_crowther_nature_2015_4326_float32.tiff"

data_paths = [
    (pet_path,"pet"),
    (td_path,"td")
]

print("Here")

for path,col in data_paths:
    raster = rasterio.open(path)
    gdf[col] = [x for x in raster.sample(coord_list)]

# Convert again to a DataFrame to explode the new values from a single-valued array to a float.
df = pd.DataFrame(gdf)
df = df.explode(
    ["pet",
     "td"]
)

# Remove Nans and 0.0 in tree density.
df = df.drop(
    df[ (df.td <= 0.0) ].index
).reset_index(drop=True)
print(df)

# Convert back to a GeoDataFrame.
gdf = gpd.GeoDataFrame(df).reset_index(drop=True)
gdf["pet"] = gdf['pet'].astype(float)
gdf["td"] = gdf['td'].astype(float)
gdf["tcq"] = gdf['tcq'].astype(float)
print(gdf)    
gdf = gdf.set_crs("EPSG:4326")

gdf.to_file("raster_predictors_global_data2.shp", driver = "ESRI Shapefile")

print("Raster sampling done, now joining biogeographic realm.")

# Load BGR polygon layer
bgr = gpd.read_file("./biome_data/Ecoregions2017.shp")
bgr = bgr[["REALM","geometry"]]
bgr = bgr.rename({"REALM":"bgr"},axis="columns")
bgr["geometry"] = bgr.geometry.to_crs("EPSG:4326")

predictors = gpd.sjoin(gdf, bgr).reset_index(drop=True).drop("index_right",axis="columns")

predictors = predictors[["geometry","x","y","tcq","pet","td","bgr"]]
print(predictors)

predictors_shp = predictors.drop(["x","y"],axis="columns")
predictors_shp.to_file("./predictor_global_data2.shp", driver = "ESRI Shapefile")
predictors_csv = pd.DataFrame(predictors.drop(["geometry"],axis="columns"))
predictors_csv.to_csv("./predictor_global_data2.csv",index=False)

# dataset_shp = gpd.read_file("./predictor_global_data.shp")
# print(dataset_shp.dtypes)
# dataset_shp[['tcq', 'pet', 'td']] = dataset_shp[['tcq', 'pet', 'td']].astype(float)

# # dataset_shp = dataset_shp.drop(
# #     dataset_shp[ (dataset_shp.tcq < -280.0) | (dataset_shp.td <= 0.0) | (dataset_shp.bgr == np.nan)].index
# # ).reset_index(drop=True)
# # print(dataset_shp)

# dataset_shp = dataset_shp.drop(
#     dataset_shp[ (dataset_shp.bgr == "N/A")].index
# ).reset_index(drop=True)
# print(dataset_shp)
# print()

# dataset_shp = dataset_shp.drop(
#     dataset_shp[ (dataset_shp.tcq < -280.0) ].index
# ).reset_index(drop=True)
# print(dataset_shp)
# print()

# # dataset_shp = dataset_shp.drop(
# #     dataset_shp[ (dataset_shp.td < 0.0) ].index
# # ).reset_index(drop=True)
# # print(dataset_shp)
# # print()


# dataset_shp.to_file("./predictor_global_data_nonan.shp", driver = "ESRI Shapefile")

# dataset_csv = pd.read_csv("./predictor_global_data.csv")
# print(dataset_csv.dtypes)
# dataset_csv[['tcq', 'pet', 'td']] = dataset_csv[['tcq', 'pet', 'td']].astype(float)
# dataset_csv = dataset_csv.drop(
#     dataset_csv[ (dataset_csv.tcq < -280.0) | (dataset_csv.td <= 0.0) | (dataset_csv.bgr == np.nan)].index
# ).reset_index(drop=True)
# print(dataset_csv)
# dataset_csv.to_csv("./predictor_global_data_nonan.csv",index=False)


# with rasterio.open("./tree_density_data/tree_density_biome_based_model_crowther_nature_2015_4326_float32.tiff") as dataset:

#     # resample data to target shape
#     data = dataset.read(
#         out_shape=(
#             dataset.count,
#             int(dataset.height * 0.1),
#             int(dataset.width * 0.1)
#         ),
#         resampling=Resampling.bilinear
#     )

#     # scale image transform
#     transform = dataset.transform * dataset.transform.scale(
#         (dataset.width / data.shape[-1]),
#         (dataset.height / data.shape[-2])
#     )

#     # Extract raster data and coordinates in a numpy array.
#     data = data[0,:,:]
#     predictor = []
#     for row,vec in enumerate(data):
#         for col,elem in enumerate(vec):
#             xy = transform * (row,col)
#             predictor.append([elem, xy[0], xy[1]])
#     predictor = np.array(predictor)
#     print(predictor.shape)

# # Create a DataFrame from the array.
# df = pd.DataFrame(predictor, columns=["td","y","x"])
# print(df)

# # Shift coordinates to match EPSG 4326
# dy = -96.374
# dx = -120.509

# df["y"] = df.y + dy
# df["x"] = df.x + dx

# # Create a GeoDataFrame using the extracted coordinates. 
# gdf = gpd.GeoDataFrame(
#     df, geometry=gpd.points_from_xy(df.x, df.y)
# ).set_crs("EPSG:4326")

# print(gdf)