"""
Creates the database using the file databaseConstructor.py
"""

## Creating a database!

from data_cleaning import *
## Note the file naming
## Number _53_ for Washington
## Number _41_ for Oregon
census_shapefile = 'Tanzania_Village_EA_2002_region.shp'
country_code = 'TZ'
year = '2002'

channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7', 'B8', 'nighttime']
sat_folder_loc = '../data/satellite/'
census_folder_loc = '../data/shapefiles/tz/enumeration_areas/'
save_folder_loc = '../data/keras/'

# a file size of 10 ~ 1mb
sample_rate = 0.4				# number of total images sampled
obs_size = 32					# size of image
processes = 2             # number of CPU cores
step = 8                    # size of step in image creation

if __name__ == "__main__":
    print 'Starting database construction'
    db = database_constructor(census_folder_loc, census_shapefile,
            sat_folder_loc, save_folder_loc, country_code, 
            year, channels, sample_rate, obs_size,
            processes, step)
    db.import_sat_image()
    db.import_census_data()
    db.join_sat_census()
    db.sampling()
    db.sample_generator_sat()
    db.save_files()
    print 'Database constructed'
    print 'Good job!'
