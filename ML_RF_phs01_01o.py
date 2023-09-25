###############################################################################################################
# ML_RF_phs01_01.py

# Created by: Robson Rogério Dutra Pereira on 10.Sep.2023
# Last Modified: rrdpereira

# Description: Crop original image and use the ground truth shapefile
             # to split the train and test samples and, use the image segmentation with quickshift and slic algorithm.
             # Realize the classification using the sklearn Random Forest.

# E-mail: robsondutra.pereira@outlook.com
###############################################################################################################
import sys, time, os, datetime, glob
from osgeo import gdal,ogr
import numpy as np
import geopandas as gpd
import pandas as pd
from skimage import exposure
from skimage.segmentation import quickshift, slic
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from platform import python_version
print(f"(Sys version) :|: {sys.version} :|:")
os.system("which python")
print(f"(Python version) :#: {python_version()} :#:")
###############################################################################################################
out_classified_timestamp = 0 # 1: ON and 0: OFF
fracPerV = 0.7
n_segmentsV = 25000 # number of segments for the segmentation multiple options with quickshift and slic
compactnessV = 0.1 # compactness for the segmentation multiple options with quickshift and slic
root_folder = "./NAIP/"
os.makedirs(root_folder, exist_ok=True)

in_image = root_folder + "m_4211161_se_12_1_20160624.tif" # original image
shp_poly = root_folder + "CropOriginalSHP.shp" # shapefile for clip (crop)
gpkg_poly = root_folder + "CropOriginalGPKG.gpkg" # gpkgfile for clip (crop)
out_shp_cropped_img = root_folder + "m_4211161_se_12_1_20160624_CROP_shp.tif" # output cropped image with shapefile
out_gpkg_cropped_img = root_folder + "m_4211161_se_12_1_20160624_CROP_gpkg.tif" # output cropped image with gpkgfile
out_segments = root_folder + "SegmentsFinal.tif"
train_shp_file = root_folder + "Train.shp"
test_shp_file = root_folder + "Test.shp"
class_lookup_csv = root_folder + "ClassLookup.csv"
out_classified_file = root_folder + "Classified.tif"

# Crop original image
# Mode 02 with SHP
if (not os.path.exists(out_shp_cropped_img)):
    print("------------>>>'Crop original image with SHP' processing!")
    gdal.Warp(out_shp_cropped_img, in_image, cutlineDSName=shp_poly, cropToCutline=True)
else:
    print("------------>>> Skipping 'Crop original image with SHP' processing, files already exist!")

# Mode 02 with GPKG
if (not os.path.exists(out_gpkg_cropped_img)):
    print("------------>>>'Crop original image with GPKG' processing!")    
    gdal.Warp(out_gpkg_cropped_img, in_image, cutlineDSName=gpkg_poly, cropToCutline=True)
else:
    print("------------>>> Skipping 'Crop original image with GPKG' processing, files already exist!")

# Load cropped image
in_cropped_image = root_folder + "m_4211161_se_12_1_20160624_CROP_shp.tif"
print('in_cropped_image: {0}'.format(in_cropped_image))

driver_tiff = gdal.GetDriverByName('GTiff')
print('driver_tiff: {0}'.format(driver_tiff))

in_image_ds = gdal.Open(in_cropped_image)
print('in_image_ds: {0}'.format(in_image_ds))
print('RasterCount: {0}'.format(in_image_ds.RasterCount))
nrows = in_image_ds.RasterYSize
ncols = in_image_ds.RasterXSize

# Check and load the image bands (channels)
n_bands = in_image_ds.RasterCount
print('n_bands: {0}'.format(n_bands))

band_data = []
print('band_data: {0}'.format(band_data))
print('Bands: {0}; Rows: {1}; Columns: {2};'.format(in_image_ds.RasterCount,in_image_ds.RasterYSize,in_image_ds.RasterXSize))
# Get raster band as array
for i in range(1, n_bands+1):
    band = in_image_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
print('band_data shape: {0}'.format(band_data.shape))

# Image Segmentation Section
def segment_features(segment_pixels):
    features = []
    # fuction_start = time.time()
    # print('function start in: {0} seconds'.format(fuction_start))
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        # print('band_stats: {0}'.format(band_stats))    
        features += band_stats
    # print('function complete in: {0} seconds'.format(time.time() - fuction_start))
    return features

# Scale image values from 0.0 - 1.0
img = exposure.rescale_intensity(band_data)
print('img shape: {0}'.format(img.shape))
print('Loaded the rescales intesisty')

# Config the segmentation multiple options with quickshift and slic
seg_start = time.time() # START check the process time
print('segments start in: {0} seconds'.format(seg_start))
segments = slic(img, n_segments=n_segmentsV, compactness=compactnessV)
print('segments complete in: {0} seconds'.format(time.time() - seg_start)) # END check the process time

# Run the segmentation multiple options with quickshift and slic
print("------------>>>'Run the segmentation multiple options with quickshift and slic' processing!")    
obj_start = time.time()
print('objects start in: {0} seconds'.format(obj_start))
segment_ids = np.unique(segments)
objects = []
object_ids = []
for id in segment_ids:
    segment_pixels = img[segments == id]
    print('pixels for id {0} and segment_pixels shape {1}'.format(id,segment_pixels.shape))
    object_features = segment_features(segment_pixels)
    # print('objects.append')
    objects.append(object_features)
    # print('object_ids.append')
    object_ids.append(id)

obj_end = time.time() - obj_start
print('Create {0} objects with {1} variables; Complete task in: {2} seconds| {3} minutes;'.format(len(objects), len(objects[0]), obj_end, (obj_end/60)))

# save segments to raster
out_segments = root_folder + "SegmentsFinal__"+time.strftime("_%Y%m%d_%H%M%S")+".tif"
print('out_segments: {0}'.format(out_segments))
segments_ds = driver_tiff.Create(out_segments, in_image_ds.RasterXSize, in_image_ds.RasterYSize,
                                1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(in_image_ds.GetGeoTransform())
segments_ds.SetProjection(in_image_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None    

# Load ground truth shapefile with geopandas
gdf = gpd.read_file(root_folder + "GroundTruth.shp")
print('gdf: {0}'.format(gdf))
# Get ground truth classes names
class_names = gdf['label'].unique()
print('class_names: {0}'.format(class_names))
class_ids = np.arange(class_names.size) + 1
print('class_ids: {0}'.format(class_ids))

# Check loaded dataset with geopandas
print('gdf no ids: {0}'.format(gdf.head()))
gdf['id'] = gdf['label'].map(dict(zip(class_names, class_ids)))
print('gdf ids: {0}'.format(gdf.head()))

# Split the train and test dataset
gdf_train = gdf.sample(frac=fracPerV)
gdf_test = gdf.drop(gdf_train.index)
print('gdf shape: {0}; training shape: {1}; test shape: {2};'.format(gdf.shape,gdf_train.shape,gdf_test.shape))
# Create the train and test shapefile
# Train
if (not os.path.exists(train_shp_file)):
    print("------------>>>'Create the train shapefile' processing!")
    gdf_train.to_file(root_folder + "Train.shp")
else:
    print("------------>>> Skipping 'Create the train shapefile' processing, files already exist!")
# Test
if (not os.path.exists(test_shp_file)):
    print("------------>>>'Create the test shapefile' processing!")
    gdf_test.to_file(root_folder + "Test.shp")
else:
    print("------------>>> Skipping 'Create the test shapefile' processing, files already exist!")

# Create lookup table with pandas data frame
df = pd.DataFrame({'label': class_names, 'id': class_ids})
if (not os.path.exists(class_lookup_csv)):
    print("------------>>>'Create lookup table' processing!")
    df.to_csv(root_folder + "ClassLookup.csv")
else:
    print("------------>>> Skipping 'Create lookup table' processing, files already exist!")

# Load points file to use for training data
train_fn = root_folder + "Train.shp"
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
# create a Gdal dataset (ds)
target_dsTr = driver.Create('', in_image_ds.RasterXSize, in_image_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_dsTr.SetGeoTransform(in_image_ds.GetGeoTransform())
target_dsTr.SetProjection(in_image_ds.GetProjection())

# rasterize the training points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_dsTr, [1], lyr, options=options)

# retrieve the rasterized data and print basic stats
data = target_dsTr.GetRasterBand(1).ReadAsArray()
print('min: {0}; max: {1}; min: {2};'.format(data.min(),data.max(),data.mean()))

# Get variables from raterized dataset (ds)
ground_truth = target_dsTr.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]
print('class values: {0}'.format(classes))

segments_per_class = {}

for ccllass in classes:
    segments_of_class = segments[ground_truth == ccllass]
    segments_per_class[ccllass] = set(segments_of_class)
    print('Training segments for class {0}; Length: {1}'.format(ccllass,len(segments_of_class)))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    #"|=" means "or equals"
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"

# Preparing training images from segments
train_img = np.copy(segments)
threshold = train_img.max() + 1

for ccllass in classes:
    class_label = threshold + ccllass
    for segment_id in segments_per_class[ccllass]:
        # "[]" where train_img == segment_id
        train_img[train_img == segment_id] = class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for ccllass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[ccllass]]
    training_labels += [ccllass] * len(class_train_object)
    training_objects += class_train_object
    print('Training objects for class {0}; Length: {1}'.format(ccllass,len(class_train_object)))

# Training images from segments with RandomForestClassifier
classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print('Fitting Random Forest Classifier')
predicted = classifier.predict(objects)
print('Predicting Classifications')
Predicting_start = time.time()
print('START Predicting in: {0} seconds'.format(Predicting_start))

clf = np.copy(segments)
for segment_id, ccllass in zip(segment_ids, predicted):
    clf[clf == segment_id] = ccllass

Predicting_end = time.time() - Predicting_start
print('END Complete Predicting in: {0} seconds| {1} minutes'.format(Predicting_end, (Predicting_end/60)))

print('Prediction applied to numpy array')
mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0

print('Saving classificaiton to raster with gdal')
if out_classified_timestamp == 1:
    clfds = driver_tiff.Create(root_folder + "Classified__"+time.strftime("_%Y%m%d_%H%M%S")+".tif",
                               in_image_ds.RasterXSize, in_image_ds.RasterYSize,
                               1, gdal.GDT_Float32)
    clfds.SetGeoTransform(in_image_ds.GetGeoTransform())
    clfds.SetProjection(in_image_ds.GetProjection())
    clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clfds.GetRasterBand(1).WriteArray(clf)
    clfds = None
    print('Random Forest Classification is Done!')

elif out_classified_timestamp == 0:
    print("------------>>>'Random Forest Classification Classified.tif file' processing!")
    out_classified_file = root_folder + "Classified.tif"
    clfds = driver_tiff.Create(out_classified_file,
                            in_image_ds.RasterXSize, in_image_ds.RasterYSize,
                            1, gdal.GDT_Float32)
    clfds.SetGeoTransform(in_image_ds.GetGeoTransform())
    clfds.SetProjection(in_image_ds.GetProjection())
    clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clfds.GetRasterBand(1).WriteArray(clf)
    clfds = None
    print('Random Forest Classification Classified.tif file is Done!')

# Load points file to use for test data
test_fn = root_folder + "Test.shp"
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()

# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
# create a Gdal dataset (ds)
target_dsTe = driver.Create('', in_image_ds.RasterXSize, in_image_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_dsTe.SetGeoTransform(in_image_ds.GetGeoTransform())
target_dsTe.SetProjection(in_image_ds.GetProjection())

# rasterize the test points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_dsTe, [1], lyr, options=options)

# retrieve the rasterized data and print basic stats
data = target_dsTe.GetRasterBand(1).ReadAsArray()
print('min: {0}; max: {1}; min: {2};'.format(data.min(),data.max(),data.mean()))

# Generate the Confusion Matrix
truth = target_dsTe.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open(root_folder + "Classified.tif")
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

cm = metrics.confusion_matrix(truth[idx], pred[idx])

# pixel based accuracy
print(cm)

print(cm.diagonal())
print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)