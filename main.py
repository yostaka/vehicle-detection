import glob
import time

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from CarND.lesson_functions import *
import CarND.visualize as vis

from scipy.ndimage.measurements import label

# Configurations
test_imgs_output_folder = 'output_images/test_images/'
test_imgs = glob.glob('./test_images/test*.jpg')
cars_imgs = glob.glob('./train_data/vehicles/**/*.png')
notcars_imgs = glob.glob('./train_data/non-vehicles/**/*.png')
sample_size = 5000

generateVideo = True
# video_input = 'test_video.mp4'
# video_output = 'output_images/video_output/test.mp4'
video_input = 'project_video.mp4'
video_output = 'output_images/video_output/car_detection.mp4'

runSampleHOGFeatureExtraction = False

if runSampleHOGFeatureExtraction is True:
    # Extract HOG features from training images
    sample_car_img = mpimg.imread('./train_data/vehicles/GTI_MiddleClose/image0190.png')
    sample_notcar_img = mpimg.imread('./train_data/non-vehicles/GTI/image18.png')

    vis.visualize(imgs=[sample_car_img, sample_notcar_img],
                  titles=['Car', 'Not-Car'],
                  fname=test_imgs_output_folder + 'car-notcar-image.jpg')

    # Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
    # and train a classifier Linear SVM classifier
    print('Performing HOG feature extraction...')
    for idx, fname in enumerate(test_imgs):
        print('  Processing', fname)
        image = mpimg.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features, hog_img = get_hog_features(gray, orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)

        vis.visualize(imgs=[image, hog_img],
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
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Visualize feature extractions
sample_car_img1 = mpimg.imread(cars[0])
sample_notcar_img1 = mpimg.imread(notcars[0])
sample_car_img2 = mpimg.imread(cars[1])
sample_notcar_img2 = mpimg.imread(notcars[1])
sample_car_img3 = mpimg.imread(cars[2])
sample_notcar_img3 = mpimg.imread(notcars[2])
vis.visualize(imgs=[sample_car_img1, sample_car_img2, sample_car_img3, sample_notcar_img1, sample_notcar_img2, sample_notcar_img3],
              titles=['Car1', 'Car2', 'Car3', 'Not-Car1', 'Not-Car2', 'Not-Car3'], ncols=3,
              fname=test_imgs_output_folder + 'car-notcar-image.jpg')

single_img_features(sample_car_img1, color_space=color_space,
                    spatial_size=spatial_size, hist_bins=hist_bins,
                    orient=orient, pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block,
                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                    hist_feat=hist_feat, hog_feat=hog_feat,
                    show_img=True,
                    fname=test_imgs_output_folder + 'sample_car')


# Feature extractions for car images and not-car images
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

# Use SVC with rbf kernel
clf = SVC(kernel='rbf', C=10)

# Check the training time for the SVC
t = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
y_pred = clf.predict(X_test)
print('Test Accuracy of SVC = ', round(accuracy_score(y_pred, y_test), 4))

# Check the prediction time for a single sample
t = time.time()


images = glob.glob('./test_images/test*.jpg')

image = mpimg.imread(images[0])
image = image.astype(np.float32)/255

# Define search window parameters
xy_windows = []
xy_overlaps = []
y_start_stops = []

xy_windows.append((64, 64))
xy_overlaps.append((0.5, 0.5))
y_start_stops.append([380, 450])

xy_windows.append((96, 96))
xy_overlaps.append((0.8, 0.8))
y_start_stops.append([380, 550])

xy_windows.append((144, 144))
xy_overlaps.append((0.8, 0.8))
y_start_stops.append([380, 650])


windows = []

for (xy_window, xy_overlap, y_start_stop) in zip(xy_windows, xy_overlaps, y_start_stops):
    window = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                          xy_window=xy_window, xy_overlap=xy_overlap)
    windows.extend(window)


window_img = draw_windows(image, windows)
vis.visualize(imgs=[window_img], titles=['windows'], fname=test_imgs_output_folder+'windows.jpg')


def process_image(img, show_img=False):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    draw_image = np.copy(img)
    labeled_image = np.copy(img)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    img = img.astype(np.float32)/255
    #
    # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                        xy_window=(140, 140), xy_overlap=(0.2, 0.2))

    hot_windows = search_windows(img, windows, clf, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    if show_img is True:
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        plt.figure()
        plt.imshow(window_img)
        plt.show()

    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 2)
    labels = label(heat)

    if show_img is True:
        plt.imshow(labels[0], cmap='gray')
        plt.show()

    labeled_image = draw_labeled_bboxes(labeled_image, labels)

    if show_img is True:
        plt.imshow(labeled_image)
        plt.show()

    return labeled_image


for fname in images:
    image = mpimg.imread(fname)
    process_image(image, show_img=True)


# Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
# and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles


# Estimate a bounding box for vehicles detected


# Build video clip with car detection

if generateVideo is True:
    clip1 = VideoFileClip(video_input)
    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(video_output, audio=False)

print("Completed all processes")
