"""

A script to create a database of images from the LANDSAT 7 database

Saves the data as a hdf5 file

"""
import numpy as np
import pandas as pd
import pickle
import h5py
from osgeo import gdal, ogr
import pyproj
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
from multiprocessing import Pool
import parmap
import os, sys


def point_wrapper(x):
    """
    A wrapper to use the point function in parallel 
    """
    return Point(x[0], x[1])

def adder(x):
    """
   
    Wrapper for parallelisation

    Takes the north west corner of an image and returns the centroid
    
    """
    return x + 16 

class validation_constructor:
    """

    A class to take a 
        - population (shp)
        - estimates of population (array)  
    And return smaller images (arrays) for use in Keras

    """
    def __init__(
            self, census_folder_loc, census_shapefile,
            est_folder_loc, estimates, locations,
            save_folder_loc, country_code, 
            year, obs_size, processes):
        """
        Initialises the class
        Inputs:
            (string)
            census_folder_loc: location of data
            census_shapefile: name of shapefile 
            est_folder_loc: location of estimates 
            save_folder_loc: location of folder to save data 
            country_code: The state name 
            year: year of investigation string  
            obs_size: the length/width of each observation 
        """
        self.census_folder_loc = census_folder_loc
        self.census_shapefile = census_shapefile
        self.est_folder_loc = est_folder_loc
        self.estimates = estimates
        self.locations = locations
        self.save_folder_loc = save_folder_loc
        self.country_code = country_code
        self.year = year
        self.obs_size = obs_size
        self.processes = processes
        if not os.path.exists( self.census_folder_loc + '/' 
                                + self.census_shapefile ):
            sys.exit('Error: File ' + self.census_folder_loc + '/' 
                    + self.census_shapefile + ' does not exist')
        # testing whether files exist
        if not os.path.exists( self.est_folder_loc + self.estimates):
                sys.exit('Error: File ' + self.est_folder_loc + self.estimates 
                        + ' does not exist')
        
    def spatialIndex(self, blocks):
        """
        Input:
            (array of shapely polygons) blocks
        Returns:
            (rtree index) idx
        """
        idx = index.Index()
        for count, block in enumerate(blocks):
            idx.insert(count, block.bounds)
        return idx

    def point_within_polygon(self, points, idx, polygons, pop):
        """
        Finds the census tract containing pixels
        Inputs:
            (rtree spatial index instance) idx
            (array shapely points) points
            (array shapely polygons) polygons
            (array of population estimates) pop
        Returns:
            (geodataframe) points, polycounts and population 
        """
        pixelPoint_db = []
        for pixel in points:
            temp_polygon = None
            temp_pop = None
            temp_slkm = None
            temp_slid = None
            temp_slnm = None
            temp_lid = None
            temp_lname = None
            for j in idx.intersection((pixel.x, pixel.y)):
                if pixel.within(polygons[j]):
                    temp_polygon = polygons[j]
                    temp_pop = pop[j]
                    temp_slkm = self.df_census['area'][j]
                    temp_slid = self.df_census['sublocation_id'][j]
                    temp_slnm = self.df_census['sublocation_name'][j]
                    temp_lid = self.df_census['location_id'][j]
                    temp_lname = self.df_census['location_name'][j]
                    break
            pixelPoint_db.append(
                    [temp_polygon, temp_pop, pixel.x, pixel.y, temp_slkm,
                        temp_slid, temp_slnm, temp_lid, temp_lname])
        return GeoDataFrame(pixelPoint_db)

 
    def import_estimates(self, placeholder=True):
        """
        Imports the cnn_estimates
        """
        pool = Pool(self.processes)
        print 'Loading estimates data'
        if placeholder:
            f = h5py.File(self.est_folder_loc + self.estimates)
            self.location = np.array(f['locations'])
            length = len(self.location)
            f.close()
            self.ests = np.random.lognormal(0, 2, length) * 100
            self.location = pool.map(Point, self.location)
        else:
            f = pickle.load(open(self.est_folder_loc + self.estimates, 'rb')) 
            self.ests = np.array(f)
            f = pickle.load(open(self.est_folder_loc + self.locations, 'rb'))
            self.location = np.array(f)
            del f
            self.location = pool.map(Point, self.location) 
        print 'Estimates loaded  (#^.^#)'

    def import_census_data(self, data2 = None):
        """
        Gets population density from census data
        Inputs:
        census_folder_loc: location of data (string)    
        census_shapefile: name of shapefile (string)
        Returns: 
            df_census: GeoDataFrame with census information
        """ 
        ## Importing shapefile 
        print 'Importing census data'
        self.df_census = GeoDataFrame.from_file(
                self.census_folder_loc + self.census_shapefile)
        print 'Census data loaded'
        self.df_census['pop'] = self.df_census['SUM_TOTAL']
        self.df_census['area'] = self.df_census['AREA_KM2']
        self.df_census['sublocation_id'] = self.df_census['SLID']
        self.df_census['sublocation_name'] = self.df_census['SLNAME']
        self.df_census['location_id'] = self.df_census['LOCID']
        self.df_census['location_name'] = self.df_census['LOCNAME']
        print '*\o/* Census data collected *\o/*'

    def join_est_census(self):
        """
        
        Combines satellite and urban database construction
        
        """
        print 'Joining satellite and census data'
        self.census_blocks = np.array(self.df_census['geometry'])
        self.census_pop = np.array(self.df_census['pop'])
        self.idx = self.spatialIndex(self.census_blocks)
        pixel_point = self.point_within_polygon(
                self.location, self.idx, self.census_blocks, self.census_pop)
        pixel_point.columns = ['poly', 'pop', 'latitude', 'longitude', 'area',
                'sublocation_id', 'sublocation_name', 'location_id', 
                'location_name']
        self.df_estimates = pixel_point
        del pixel_point
        self.df_estimates['estimates'] = self.ests
        self.df_estimates.to_csv(self.save_folder_loc + 
                'estimates_after_join.csv')
        print 'Spatial join complete o/\o'
    

