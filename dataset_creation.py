"""
This module includes functions used to build a dataset by:
    1 - adding features from raster data.
    2 - adding features from vector data.

The dataset must be a GeoDataFrame describing Point geometries.
Raster and/or vector data is loaded in GeoDataFrames before addition.

Created on Monday January 02 2023.
@author: Diego Bengochea Paz.
"""

import geopandas as gpd
import pandas as pd

import numpy as np

import rasterio

def add_feature_from_raster(data, feature_name, feature_data, dtype):
    """
    This function adds a new feature to a dataset by intercepting coordinates of points in the dataset 
    with the tiles of the raster describing the new feature.
    :param data: the current Point dataset in GeoDataFrame format. 
    :param feature_name: the name of the feature to be added.
    :param feature_data: the path to the feature data.
    :param dtype: the data type of the new feature.
    :return: a GeoDataFrame incorporating the new feature. 
    """

    # Extract coordinates from the geometry column of the dataset.
    coordinates = [(x,y) for x,y in zip(data['geometry'].x , data['geometry'].y)]

    # Open the new feature's raster and add the new feature to the dataset by sampling the coordinates.
    raster = rasterio.open(feature_data)
    data[feature_name] = [x for x in raster.sample(coordinates)]

    # Aesthetic changes.
    df = pd.DataFrame(data).explode( [feature_name] )

    return gpd.GeoDataFrame(df).reset_index(drop=True).astype({feature_name:dtype})

def add_feature_from_polygon_layer(data,feature_name,feature_data,col_name):
    """
    This function adds a new feature to a Point dataset by performing a spatial join with a Polygon layer describing the new feature.
    :param data: the current Point dataset in GeoDataFrame format.
    :param feature_name: the name of the feature to be added in the polygon layer.
    :param feature_data: the path to the new feature's polygon layer.
    :param col_name: the name of the new feature in the dataset. 
    TODO: the polygon layer may have tons of unuseful information that we may want to discard.
    """

    # Project the dataset's geometry to 4326.
    data["geometry"] = data.geometry.to_crs("EPSG:4326")
    
    # Load the polygon layer containing the data on the new feature.
    polygons = gpd.read_file(feature_data).rename({feature_name:col_name},axis="columns")

    # Perform a spatial join between the dataset and the polygon layer.
    return gpd.sjoin(data, polygons).reset_index(drop=True).rename({"index_right":col_name+"_id"},axis="columns")

