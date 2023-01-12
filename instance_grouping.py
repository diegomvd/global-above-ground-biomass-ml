"""
This module includes functions used to group instances by spatial proximity and to aggregate feature values within groups of instances.
Two grouping methods are provided: 
    1- Intersecting buffers.
    2- Grid tiles.
Dataset instances are rows of a Point GeoDataFrame.

Created on Monday January 02 2023.
@author: Diego Bengochea Paz.
"""


def group_by_intersecting_buffers(data,radius,bbox_length):
    """
    This function creates groups of instances when instances's buffer disks of specified radius intersect with each other.  
    :param data:
    :param radius:
    :param bbox_length:
    :return:
    """  

    # Re-project coordinates in Web Mercator CRS to use distances in Meters. 
    data_groups = data["geometry"].to_crs("EPSG:3857")

    # Create the buffer zones with the specified radius.
    data_groups["geometry"] = data_groups["geometry"].buffer(radius)

    # Dissolve intersecting buffers: this step looses all information on individual instances.
    data_groups = data_groups.dissolve().explode(index_parts=True).reset_index(drop=True) 

    # Assign an Id to each group of instances.
    data_groups["gid"] = range(data_groups.shape[0])

    # Select group Id and geometry columns.
    data_groups = data_groups[["gid",'geometry']]

    # Convert back to EPSG:4326
    data_groups = data_groups.to_crs("EPSG:4326")

    # Join the group information with the original datase by doing a spatial join between buffers and points.
    new_data = gpd.sjoin(data,data_groups).reset_index(drop=True).drop("index_right",axis="columns")

    # Add a column with information on number of samples in each site.
    new_data["n_samples"] = data["gid"].map(data["gid"].value_counts())

    return new_data

def group_by_tiling():
    """
    """

def aggregate_values_within_group():
    """
    """