"""
Comparison of two maps through the Structural Similarity Index (SSIM) described in 
Jones et al. 2016 "Novel application of a quantitative spatial comparison tool to 
species distribution data". 

Production of new maps with: 
- map of differences in mean value.
- map on differences in variance. 
- map of covariance structure. 
- map of general similarity index
- global similarity index in function of size of the moving window..
"""

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd


# Paths to the two rasters to compare 

raster1 = ""
raster2 = ""

# Functions for the similarity

def similarity_in_mean(mean1,mean2,val_range):
    """
    :param mean1: the window mean of raster 1.
    :param mean2: the window mean of raster 2.
    :param range: the range of values in the window for both rasters.
    :return: the window's similarity in mean.
    """

    k1 = 0.01
    c1 = np.power(k1*val_range,2)

    return (2*mean1*mean2+c1)/(mean1**2+mean2**2+c1) 

def similarity_in_variance(var1,var2,val_range):
    """
    :param var1: the window variance of raster 1.
    :param var2: the window variance of raster 2.
    :param range: the range of values in the window for both rasters.
    :return: the window's similarity in mean.
    """

    k2 = 0.03
    c2 = np.power(k2*val_range,2)

    return (2*np.sqrt(var1)*np.sqrt(var2)+c2)/(var1**2+var2**2+c2) 


def similarity_in_pattern(var1,var2,cov,val_range):
    """
    :param var1: the window variance of raster 1.
    :param var2: the window variance of raster 2.
    :param cov: the covariance between both raster windows.
    :param range: the range of values in the window for both rasters.
    :return: the window's similarity in mean.
    """

    k2 = 0.03
    c3 = 0.5*np.power(k2*val_range,2)

    return (cov+c3)/(np.sqrt(var1)*np.sqrt(var2)+c3) 

def overall_similarity(s_mean,s_variance,s_pattern,w_mean,w_variance,w_pattern): 

    return np.power(s_mean,w_mean)*np.power(s_variance,w_variance)*np.power(s_pattern,w_pattern)

# Window statistics

def window_mean(array, weights):
    """
    :param array: the numpy array corresponding to the raster.
    :param weights: the array of weights for the arithmetic mean calculation.
    :return: the weighted mean value of the window.
    """

    return np.sum(weights*array)



def window_variance(array, mean, weights):
    """
    :param array: the numpy array corresponding to the raster.
    :param weights: the array of weights for the arithmetic mean calculation.
    :param mean: the window mean.
    :return: the weighted variance of the window.
    """

    diff = array - mean
    return np.sum( weights * np.power(diff,2) )

def window_covariance(array1,array2,mean1,mean2,weights): 
    """
    :param array1: the numpy array corresponding to the raster 1.
    :param array2: the numpy array corresponding to the raster 2.
    :param mean1: the window mean of raster 1.
    :param mean2: the window mean of raster 2.
    :param weights: the array of weights for the arithmetic mean calculation.
    :return: the weighted covariance between rasters in the window.
    """       

    diff1 = array1-mean1
    diff2 = array2-mean2

    return np.sum(weights * diff1 * diff2)

# Window creation 

def create_windows():
    return

def weights():
    return