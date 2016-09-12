"""

Converting satellite images to population estimates. 

Creating data for postestimation. Using

Images. Using
- Satellite data
- Population data

Estimates. Using
- Keras weights
- Keras model

"""

from data_cleaning import *

census_shapefile = 'Tanzania_Village_EA_2002_region.shp'
country_code = 'KE'
resolution = 250 
year = '2002'

channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2',  'B8', 'lights']
sat_folder_loc = '../data/satellite/'
census_folder_loc = '../data/shapefiles/tz/enumeration_areas/'
save_folder_loc = '../data/estimates/'
save_filename = country_code + year + 'map.hdf5'

# a file size of 10 ~ 1mb
sample_rate = 1.0			# number of total images sampled
obs_size = 32				# size of image
processes = 2                           # number of CPU cores
step = 8                                # size of step in image creation

if __name__ == "__main__":
    print 'Starting database construction'
    db = database_constructor(census_folder_loc, census_shapefile,
            sat_folder_loc, save_folder_loc, country_code, 
            year, channels, sample_rate, obs_size, resolution,
            processes, step)
    db.import_sat_image()
    db.sample_generator_sat(post=True)
    db.index_to_lat_lon()
    db.save_files_postestimation()
    print 'Database constructed'
    print 'Good job!'

