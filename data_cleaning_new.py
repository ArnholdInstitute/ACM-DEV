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
    x[0] : longitude
    x[1]: latitude
    """
    return Point(x[0], x[1])

def pixel_to_coordinates(column, row, geotransform):
    """

    Returns lat lon coordinates from pixel position 
    using an affine transformation

    See http://www.gdal.org/gdal_datamodel.html

    Outside of the class because it's used in parallel

    Input:
        (array) geotransfrom
        (int) column, row
    Returns:
        (float) lat, lon projection coordinates

    """
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    rotation_x = geotransform[2]
    rotation_y = geotransform[4]
    # The affine transformation
    lon_coord = x_origin + (column * pixel_width) + (row * rotation_x)
    lat_coord = y_origin + (column * rotation_y) + (row * pixel_height)
    return (lon_coord, lat_coord)

def array_wrapper(col, row, array):
    """
    A wrapper to use the point function in parallel 
    """
    return array[row][col]

def adder(x, y):
    """
   
    Wrapper for parallelisation

    Takes the north west corner of an image and returns the centroid
    
    """
    return x + y 

class database_constructor:
    """

    A class to take a 
        - satellite image (tiff)
        - population estimates (shp)
        - urban/rural location indicator (shp)
    And return smaller images (arrays) for use in Keras

    """
    def __init__(
            self, census_folder_loc, census_shapefile,
            sat_folder_loc, save_folder_loc, country_code, 
            year, channels, sample_rate, obs_size, resolution,
            processes, step):
        """
        Initialises the class
        Inputs:
            (string)
            census_folder_loc: location of data
            census_shapefile: name of shapefile 
            urban_folder_loc: location of urban data 
            urban_shapefile: name of shapefile
            sat_folder_loc: location of satellite images 
            save_folder_loc: location of folder to save data 
            country_code: The state name 
            year: year of investigation string  
            (list of str)
            channels: bandwidths used in estimation list of strings
            (float)
            sample_rate: proportion of images sampled 
            (int)
            obs_size: the length/width of each observation 
            resolution: resolution of sat images
        """
        self.census_folder_loc = census_folder_loc
        self.census_shapefile = census_shapefile
        self.sat_folder_loc = sat_folder_loc
        self.save_folder_loc = save_folder_loc
        self.country_code = country_code
        self.year = year
        self.channels = channels
        self.sample_rate = sample_rate
        self.obs_size = obs_size
        self.resolution = resolution
        self.processes = processes
        self.step = step
        if not os.path.exists( self.census_folder_loc + '/' 
                                + self.census_shapefile ):
            sys.exit('Error: File ' + self.census_folder_loc + '/' 
                    + self.census_shapefile + ' does not exist')
        self.filename = self.sat_folder_loc + self.country_code + \
                '_' + str(self.resolution) + '_B1.tif'
        # testing whether files exist
        if not os.path.exists( self.filename ):
                sys.exit('Error: File ' + self.filename + 
                        ' does not exist')
        self.satellite_gdal = gdal.Open(self.filename)

        
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
            for j in idx.intersection((pixel.x, pixel.y)):
                if pixel.within(polygons[j]):
                    temp_polygon = polygons[j]
                    temp_pop = pop[j]
                    break
            pixelPoint_db.append([temp_polygon, temp_pop, pixel.x, pixel.y])
        return GeoDataFrame(pixelPoint_db)

    def crop_image(self):
        """
        Crops the image to remove a lot of the empty space
        """
        # getting image
        x = self.satellite_gdal.GetRasterBand(1)
        x = x.ReadAsArray() 
        # how much of array is masked
        print '%.1f%% masked' % (np.isnan(x).sum() * 100.0 / x.size)
        # cropping top off 
        i = 0
        while np.isnan(x[i,:]).sum() == len(x[i,:]): i += 1
        self.row_min = i
        # cropping off bottom
        i = len(x[:, 0]) - 1
        while np.isnan(x[i,:]).sum() == len(x[i,:]): i -= 1
        self.row_max = i
        # cropping left off 
        i = 0
        while np.isnan(x[:, i]).sum() == len(x[:, i]): i += 1
        self.col_min = i
        # cropping off bottom
        i = len(x[0, :]) - 1
        while np.isnan(x[:, i]).sum() == len(x[:, i]): i -= 1
        self.col_max = i
        del x
        print '(Lat) Row min: ', self.row_min
        print '(Lon) Col min: ', self.col_min

    def get_location(self):
        """

        Extracts the location of each pixel in the satellite image

        """
        self.ncols = self.row_max - self.row_min
        # self.satellite_gdal.RasterXSize 
        self.nrows = self.col_max - self.col_min
        # self.satellite_gdal.RasterYSize 
        self.length_df = self.nrows * self.ncols
        print 'Columns, rows', self.ncols, self.nrows
        cols_grid, rows_grid = np.meshgrid(
                    range(self.col_min, self.col_max), 
                    range(self.row_min, self.row_max))
        self.cols_grid = cols_grid.flatten()
        self.rows_grid = rows_grid.flatten()
        # getting a series of lat lon points for each pixel
        self.geotransform = self.satellite_gdal.GetGeoTransform()
        print 'Getting locations'
        self.location_series = np.array(parmap.starmap(
                        pixel_to_coordinates, 
                        zip(self.cols_grid, self.rows_grid), 
                        self.geotransform,
                        processes = self.processes))
        print 'NW corner: ', self.location_series[0]
        print 'SE corner: ', self.location_series[-1]
        print 'Converting to Points'
        pool = Pool(self.processes)
        self.location_series = pool.map(
                        point_wrapper, 
                        self.location_series)
        print 'NW corner: ', self.location_series[0]


    def image_slicer(self, image):
        """
        
        Cuts the larger satellite image into smaller images 
        A less intense version of extract_patches_2d
        
        Input:
            (2d array) satellite image or population map

        """
        self.patches = []
        self.indices = []
        if len(image.shape) == 2:
            for y in range(0, self.nrows, self.step):
                for x in range(0, self.ncols, self.step):
                    mx = min(x+self.obs_size, self.ncols)
                    my = min(y+self.obs_size, self.nrows)
                    tile = image[ y: my, x: mx ]
                    if tile.shape == (self.obs_size, self.obs_size):
                        self.patches.append(tile)
                        self.indices.append((x, y))
        else:
            for y in range(0, self.nrows, self.step):
                for x in range(0, self.ncols, self.step):
                    mx = min(x+self.obs_size, self.ncols)
                    my = min(y+self.obs_size, self.nrows)
                    tile = image[ :, y : my, x: mx ]
                    if tile.shape == (self.obs_size, self.obs_size):
                        self.patches.append(tile)
                        self.indices.append((x, y))
        self.patches = np.array(self.patches)
        self.indices = np.array(self.indices)

    def sample_extractor(self, data_array, axis):
        """

        Extracts a sample of images
            (array) data_array, satellite images
            (axis) axis of array
            (array) image_sample, Keras ready numpy array of images
        
        """
        self.image_slicer(data_array)
        self.image_sample = np.take(
                self.patches, self.sample_idx, axis=axis,
                mode = 'raise')
 
    def import_sat_image(self):
        """
        Converts satellite images to a GeoDataFrame
        The satellite image used here is the 2010 LANDSAT 7 TOA composite
        Returns:
            (geodataframe) df_image
            (int) nrows, ncols. size of satellite image
        """
        data = []
        count = 0
        self.crop_image()
        self.get_location()
        for extension in self.channels:
            self.filename = self.sat_folder_loc + self.country_code + \
                        '_' + str(self.resolution) + '_' + extension + '.tif'
            # testing whether files exist
            if not os.path.exists( self.filename ):
                sys.exit('Error: File ' + self.filename + 
                        ' does not exist')
            # getting data
            print 'Loading bandwidth', extension
            self.satellite_gdal = gdal.Open(self.filename)
            band = self.satellite_gdal.GetRasterBand(1)
            band_array = band.ReadAsArray(
                    )[self.row_min:self.row_max, 
                            self.col_min:self.col_max]
            band_array = np.nan_to_num(band_array)
            # sanity check that we're loading different channels
            print 'Mean of bandwidth: ', band_array.mean()
            data.append(band_array.flatten())
        data = np.array(data)
        self.df_image = GeoDataFrame({'location': self.location_series})
        for count, extension in enumerate(self.channels):
            self.df_image[extension] = data[count]
 
    def import_census_data(self):
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
        # It turns out the earth isn't flat 
        # Getting area in km**2 
        #print 'Calculating area'
        #area_sq_degrees = self.df_census['geometry'] 
        #area_sq_km = [] 
        #for region in area_sq_degrees: 
        #    geom_area = ops.transform(
        #            partial(
        #                pyproj.transform,
        #                # projection GSCNAD83
        #                # southern WA EPSG:2286
        #                pyproj.Proj(init='EPSG:4326'), 
        #                pyproj.Proj(
        #                    proj='aea', 
        #                    lat1=region.bounds[1], 
        #                    lat2=region.bounds[3]
        #                    )
        #                ), 
        #            region) 
        #    area = geom_area.area / 1000000.0  #convert m2 to km2
        #    area_sq_km.append( area )
        #self.df_census['area'] = area_sq_km
        self.df_census['density'] = self.df_census['POPDENS']
        #        self.df_census['POP10'] / self.df_census['area']
        print 'Area calculated'

    def join_sat_census(self):
        """
        
        Combines satellite and urban database construction
        
        """
        print 'Joining satellite and census data'
        self.census_blocks = np.array(self.df_census['geometry'])
        self.census_pop = np.array(self.df_census['density'])
        self.idx = self.spatialIndex(self.census_blocks)
        pixel_point = self.point_within_polygon(
                self.location_series, self.idx, self.census_blocks,
                self.census_pop)
        pixel_point.columns = ['poly', 'pop', 'latitude', 'longitude']
        self.df_image['pop_density'] = pixel_point['pop']
        del pixel_point
        self.df_image.to_csv('after_join.csv')
    
    def sampling(self):
        """
        Constructs a weighted sample of images from the GeoDataFrame
        
        Returns:
            (array) sample_idx: index of sampled images 
        
        Note: Keras uses the last x percent of data in cross validation 
            Have to shuffle here to ensure that the last ten percent isn't just
            the southern most rows of information
        """
        # Getting the sum of urban pixels for each patch
        self.pop_array = self.df_image['pop_density'].fillna(0)
        self.pop_array = np.array(
                self.pop_array).reshape((self.nrows, self.ncols))
        print 'Extracting population patches'
        self.image_slicer(self.pop_array)
        pop_count = np.array([np.mean(patch) for patch in self.patches])
        self.df_sample = pd.DataFrame(pop_count, columns=['pop_ave'])
        #print 'get locations for individual frames'
        #pool = Pool(self.processes)
        #cols_grid = pool.map(adder, self.indices[:,0])
        #rows_grid = pool.map(adder, self.indices[:,1])
        #print 'Max of cols grid after slicing:',  max(cols_grid)
        #print 'Max of rows grid after slicing:',  max(rows_grid)
        #self.frame_location_series = parmap.starmap(
        #        pixel_to_coordinates,
        #        zip(cols_grid, rows_grid), self.geotransform, 
        #        processes=self.processes)
        #print 'converting locations to Points'
        #self.frame_location_series = \
        #        pool.map(Point, self.frame_location_series)
        ## Getting the locations
        #self.df_sample['location'] = self.frame_location_series
        # Creating sample weights 
        seed  = 1975
        self.pop_output_data = self.df_sample.sample(
                frac=self.sample_rate,
                replace=True,
                weights='pop_ave',
                random_state = seed)
        self.sample_idx = np.array(self.pop_output_data.index.values)
        self.pop_output_data = np.array(
                self.pop_output_data).reshape((
                    len(self.pop_output_data), 1))
    
    def sample_generator_sat(self, post):
        """
        
        Constructs a sample of observations that Keras can play with 
        
        """
        # satellite image data
        image_array = []
        if post:
            for channel in self.channels:
                tmp_img = np.array(
                        self.df_image[channel]).reshape((self.nrows, self.ncols))
                self.image_slicer(tmp_img[:,:])
                image_array.append(np.array(self.patches))
        else: 
            for channel in self.channels:
                tmp_img = np.array(
                        self.df_image[channel]).reshape((self.nrows, self.ncols))
                self.sample_extractor(tmp_img[:,:], axis=0)
                image_array.append(np.array(self.image_sample))
        image_array = np.array(image_array)    
        image_array = np.swapaxes(image_array, 0, 1)
        self.image_output_data = np.array(image_array)

    def index_to_lat_lon(self):
        """
        Converts the index of the upper left corner of the image
        to the lat lon of the 'middle' pixel 
        """
        pool = Pool(self.processes)
        cols_grid = parmap.map(adder, self.indices[:,0], 
                16 + self.col_min, processes=self.processes)
        rows_grid = parmap.map(adder, self.indices[:,1],
                16 + self.row_min, processes=self.processes)
        self.post_location_series = parmap.starmap(
                pixel_to_coordinates,
                zip(cols_grid, rows_grid), self.geotransform, 
                processes=self.processes)

    def save_files(self):
        """
        Saves the image information
        Inputs:
        X: numpy array of image samples
        save_folder_loc: location to save data
        country_code: state you're sampling 
        Returns:
        Nothing, it just saves the data! 
        """
        print 'Image output shape: ', self.image_output_data.shape
        f = h5py.File(self.save_folder_loc + 'db_' +
                 self.country_code + '_' + str(self.resolution) + '.hdf5', 'w')
        f.create_dataset('features', 
                data = self.image_output_data, 
                compression="gzip")
        f.create_dataset('targets',
                data = self.pop_output_data.tolist(), 
                compression="gzip")
        f.close()


    def save_files_postestimation(self):
        """
        Saves the images for use in the keras model
        """
        print 'Image output shape: ', self.image_output_data.shape
        f = h5py.File(self.save_folder_loc + 'db_post_' +
                 self.country_code + '_' + str(self.resolution) + '.hdf5', 'w')
        f.create_dataset('features', 
                data = self.image_output_data, 
                compression="gzip")
        f.create_dataset('locations', 
                data = self.post_location_series, 
                compression="gzip")

        f.close()

