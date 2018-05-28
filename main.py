import glob
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from skimage.feature import hog
from CarND.lesson_functions import *
import CarND.visualize as vis

# Configurations
test_imgs_output_folder = './output_images/test_images/'
test_imgs = glob.glob('./test_images/test*.jpg')
cars_imgs = glob.glob('./train_data/vehicles/**/*.png')
notcars_imgs = glob.glob('./train_data/non-vehicles/**/*.png')
sample_size = 500


# Extract HOG features from training images
sample_car_img = mpimg.imread('./train_data/vehicles/GTI_MiddleClose/image0190.png')
sample_notcar_img = mpimg.imread('./train_data/non-vehicles/GTI/image18.png')

vis.visualize(imgs=[sample_car_img, sample_notcar_img],
              titles=['Car', 'Not-Car'],
              fname=test_imgs_output_folder + 'car-notcar-image.jpg')


# Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
# and train a classifier Linear SVM classifier
print('Performing HOG feature extraction...')
for idx, fname in enumerate(test_imgs):
    print('  Processing', fname)
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features, hog_img = get_hog_features(gray, orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)

    vis.visualize(imgs=[img, hog_img],
                  titles=['Original: ' + fname.split('/')[-1],
                   'HOG feature extraction'],
                  cmaps=[None, 'gray'],
                  fname=test_imgs_output_folder + fname.split('/')[-1].split('.')[0] + '.jpg')



# (Optional) Apply a color transform and append binned color features, as well as histograms of color
# to HOG feature vector


# Search for vehicles in images by sliding-window technique with the trained classifier
# print('Searching for vehicles by sliding-window...')
# for idx, fname in enumerate(test_imgs):

# Read in cars and not cars
cars = []
notcars = []
for img_fname in cars_imgs:
    cars.append(img_fname)

for img_fname in notcars_imgs:
    notcars.append(img_fname)

# Reduce the sample size
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]









# Configuration parameters for extracting features
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [450, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()


image = mpimg.imread('./test_images/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.figure()
plt.imshow(window_img)
plt.show()

# Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
# and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles


# Estimate a bounding box for vehicles detected


print("Completed all processes")
