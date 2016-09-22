"""

Combining two kenyan shapefiles to make a single better one!

The first shapefile has sublocations -- a low administrative level
The second shapefile has locations -- with additional information on districts
and regions

Combining the two so we can merge with population data

"""
import fiona 
import shapely.geometry 
import geopandas as gpd

file_loc = '../data/shapefiles/ke/Location.shp'
file_subloc = '../data/shapefiles/ke/Ken_Sublocations.shp'

df_loc = gpd.GeoDataFrame.from_file(file_loc)
df_subloc = gpd.GeoDataFrame.from_file(file_subloc)

for i, sl in enumerate(polygons_subloc):
    point = Point(sl['geometry']['coordinates'][0]).centroid
    for j in idx.intersection(point.coords[0]):
        if point.within(shape(polygons_loc[j]['geometry'])):
            sl['properties']['LOCID'] = polygons_loc[j]['properties']['LOCID']
