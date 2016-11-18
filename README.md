# LANDSAT-landstats (ACM-DEV paper edition)

We can use the files in this repository to predict population from
satellite images.

LANDSAT lanstats is a supervised learning model to predict
socio-economic characteristics from satellite data. We can easily
modify this code to predict other socio-economic characteristics or use
other satellite images. In addition to the convnet, I've provide files to
construct the data.

The files of interest are:
 - `cnn.py`: trains the convolutional neural network
 - `cnn_estimation.py`: uses trained model for estimation
 - `data_cleaning.py`: a class to merges satellite images with population
   databases (shapefiles)
 - `do_data_cleaning.py`: runs `data_cleaning.py` to create training dataset
 - `validation.py`: class to create validation dataset
 - `do_validation.py`: creates validation data set
 - `do_postestimation.py`: converts satellite images to predictions 
 - `estimates.ipynb`: generating estimates from predictions 
 
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


