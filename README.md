# LANDSAT-landstats (ACM-DEV paper edition)

We can use the files in this repository to predict population from
satellite images.

LANDSAT lanstats is a supervised learning model to predict
socio-economic characteristics from satellite data. We can easily
modify this code to predict other socio-economic characteristics or use
other satellite images. In addition to the convnet, I've provide files to
construct the data.

The files of interest are:
 - `cnn.py`: runs the convolutional neural network
 - `data_cleaning.py`: a class to merges satellite images with population
   databases (shapefiles)
 - `data_cleaning_validation.py`: a class to turn satellite images into keras
   ready data
 - `ee_data.py`: downloads the satellite images   
 - `do_training.py`: creates training dataset 
 - `do_validation.py`: creates validation data set
 - `do_postestimation.py`: converts satellite images to predictions 
 

In addition to the usual, the analysis relies on these packages 

- [Keras](http://www.keras.io)
- [Google Earth Engine](https://developers.google.com/earth-engine/)
- [Shapely](http://www.toblerity.org/shapely/manual.html)
- [GeoPandas](http://www.geopandas.org/user.html)
- [Rtree](http://toblerity.org/rtree)
- [GDAL](https://pypi.python.org/pypi/GDAL/)
- [pyproj](https://github.com/jswhit/pyproj)
- [parmap](https://parmap.readthedocs.org)
- [h5py](http://www.h5py.org)




