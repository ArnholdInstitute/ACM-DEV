"""
Creates the database using the file databaseConstructor.py
"""

## Creating a database!

from validation import *

## Note the file naming
census_shapefile = 'ke_1999.shp'
estimates = 'kenyan_estimates.p'
locations = 'kenyan_locations.p'
country_code = 'KE'
year = '1999'

est_folder_loc = '../data/estimates/'
census_folder_loc = '../data/shapefiles/ke/sublocations/'
save_folder_loc = '../data/estimates/'

# a file size of 10 ~ 1mb
obs_size = 32				# size of image
processes = 2               # number of CPU cores

if __name__ == "__main__":
    print('Starting database construction')
    db = validation_constructor(census_folder_loc, census_shapefile,
            est_folder_loc, estimates, locations, 
            save_folder_loc, country_code, 
            year, obs_size, processes)
    db.import_estimates(placeholder=False)
    db.import_census_data()
    db.join_est_census()
    print('Database constructed')
    print('Good job!')
