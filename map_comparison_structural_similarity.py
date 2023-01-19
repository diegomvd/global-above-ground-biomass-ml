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
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

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
def reflect_border(raster,row,col,d_height,d_width,side_length):
    height = side_length - np.floor(side_length*0.5) + d_height
    width = side_length - np.floor(side_length*0.5) + d_width 
    
    row_origin = 
    col_origin = 
    
    window = create_window(raster,col_origin,row_origin,height,width)    
    
    reflection = np.zeros((side_length,side_length))
    for row in range(side_length):
        for col in range(side_length):
            i = np.int(np.abs(row - np.floor(side_length*0.5)))
            j = np.int(np.abs(col - np.floor(side_length*0.5)))
            reflection[row,col] = window[i,j]

    
    return 

def create_window(raster,col,row,width,height):
    return raster.read(1, window=Window(col,row,width,height))
    
def weights(side_length):
    weights = np.ones((side_length,side_length))
    std = weights.size*0.3333333
    return gaussian_filter(weights,sigma=std)

def value_range(array1,array2):
    max = np.max(np.array([array1,array2]))
    min = np.mix(np.array([array1,array2]))
    return max-min

# Window statistics

def window_sim(window1,window2,val_range,weights):
    mean1 = window_mean(window1,weights)
    mean2 = window_mean(window2,weights)
    return similarity_in_mean(mean1,mean2,val_range)

def window_siv(window1,window2,val_range,weights):
    mean1 = window_mean(window1,weights)
    mean2 = window_mean(window2,weights)
    var1 = window_variance(window1,mean1,weights)
    var2 = window_variance(window2,mean2,weights)
    return similarity_in_variance(var1,var2,val_range)

def window_sip(window1,window2,val_range,weights):
    mean1 = window_mean(window1,weights)
    mean2 = window_mean(window2,weights)
    cov = window_covariance(window1,window2,mean1,mean2,weights)
    var1 = window_variance(window1,mean1,weights)
    var2 = window_variance(window2,mean2,weights)
    return similarity_in_pattern(var1,var2,cov,val_range)

    
# Map comparison

def map_index(raster1,raster2,side_length,index):
    with rasterio.open(raster1) as src1:
        with rasterio.open(raster2) as src2:
            nrows = src1.height
            ncols = src1.width
   
            index_map = np.zeros((nrows-side_length,ncols-side_length))

            for row in range(nrows-side_length):
                for col in range(ncols-side_length):
                    window1 = create_window(src1,col,row,side_length,side_length)
                    window2 = create_window(src2,col,row,side_length,side_length)
                    
                    val_range = value_range(window1,window2)
                    weights = weights(side_length)

                    metric = index(window1,window2,val_range,weights)

                    index_map[row,col] = metric

    return index_map    

def map_sim(raster1,raster2,side_length):
    return map_index(raster1,raster2,side_length,window_sim)

def map_siv(raster1,raster2,side_length):
    return map_index(raster1,raster2,side_length,window_siv)       

def map_sip(raster1,raster2,side_length):
    return map_index(raster1,raster2,side_length,window_sip)       


            
            
