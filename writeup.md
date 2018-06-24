## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/test_images/car-notcar-image.jpg
[image2]: ./output_images/test_images/sample_car_feature_image.jpg
[image3]: ./output_images/test_images/sample_car_hog_image.jpg
[image4]: ./output_images/test_images/windows.jpg

[image5]: ./report_images/car-detection-1.png
[image6]: ./report_images/car-detection-2.png
[image7]: ./report_images/car-detection-3.png

[video1]: ./project_video.mp4
[video2]: ./output_images/video_output/car_detection.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Code Structure

Here's the code structure for this project:

```
.
+--- CarND/             # Functions built for vehicle detection
|  +--- lesson_functions.py
|  +--- visualize.py
|
+--- output_images/
|  +--- test_images/    # Image outputs
|  +--- video_output/   # Video outputs
|
+--- train_data/        # Training data for car and not-car images, and their labels
|
+--- main.py            # Main code - entry point for the vehicle detection
```


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 75 through line 88 of the file called `main.py`.

I started by reading in 5000 images for `vehicle` and `non-vehicle` images respectively.  Here is an example of three of each of the `vehicle`(shown as `Car`) and `non-vehicle` classes (shown as `Not-Car`):

![alt text][image1]

I then explored different color spaces and different Hog parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  And I picked `YCrCb` as color space and HOG parameters of `orientation=9`, `pixels_per_cell = 8` and `cells_per_block = 2`.

Here's the image showing channels in `YCrCb` color space:

![alt text][image2]


Here's the image showing HOG feature extraction results. I used `Y` channel (ch0) as input to the HOG because the image of the channel looks good to understand the shape of the car.

![alt text][image3]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I settled the following HOG parameters:

| Parameters     | Values             |
|:--------------:|:------------------:|
| Color space    | Y channel of YCrCb | 
| Orientation    | 9                  | 
| Pixel per cell | 2                  | 

I selected the YCrCb color space and Y cahannel for the HOG feature extraction because the image of Y channel looks suitable to extract characteristics of vechicle shapes. Orienation of 9 and pixel per cell is picked so I can keep the shape characteristics and also I can remove unnecessary details so the trained model can be generic. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in line 111 through line 118 of the file called `main.py`.

I trained SVM classifier using `rbf` kernel and C parameters set to 10 because it produced better accuracy in validation comparing to the one using linear kernel.

I used HOG features for `Y` channel as well as color features including bin spatial and color histogram.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in line 133 through line 210 of file `main.py`.

I decided to use multiple sizes of windows and use different overlap parameters and searching area for the windows. Followings are the three window sizes and their associated parameters. I used more overlaps for larger window sizes so that window search would shift its location to catch vehicle area. 

| Window size   | Overlap param. | Searching area (y-axis) |
|:-------------:|:--------------:| :----------------------:|
| 64 x 64       | (0.5, 0.5)     | 380 to 450              | 
| 96 x 96       | (0.8, 0.8)     | 380 to 550              | 
| 144 x 144     | (0.8, 0.8)     | 380 to 650              | 

Here's the area showing all the windows:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using Y-channel HOG features from YCrCb color space plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

I optimized my classifier to changing its kernel from linear to rbf, and selected three scaled windows.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/video_output/car_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is the recorded positions of positive detections in a frame of the video:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the image above:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the frame:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced challanging situation where many false positives are detected on video frames. The false positives includes guard rails and road with tree shadow. I mitigated the false positives by increasing number of training data set (from original 500 to 5,000), changing color space from RGB, HSV to YCrCb, and changing SVM kernel from linear to rbf.

I think I could make it more robust by introducing outlier detection using multiple adjacent frames, and add more not-car (non-vehicle) training data.

