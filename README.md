
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
[car]: ./out/car.jpg
[notcar]: ./out/notcar.jpg
[car_hog]: ./out/77.png
[notcar_hog]: ./out/76.png
[c]: ./out/car_hogy.jpg


[image4]: ./out/examples/sliding_window.jpg
[image5]: ./out/examples/bboxes_and_heat.pngotcar
[image6]: ./out/examples/labels_map.png
[image7]: ./out/examples/output_bboxes.png
[video1]: ./out/project_video.mp4


---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the in lines # through # of the file called `some_file.py`  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes and their Hog visualizations :

#### Car
![alt text][car]
#### Hog visulazation of Car 
![alt text][car_hog]
#### NotCar
![alt text][notcar]
#### Hog visulazation of Not Car 
![alt text][notcar_hog]


The `extract_features` function is called for the list of images, in lines `127 through 179` in `project5.py`. It extracts Hog features, spatial and color histogram features(I have only used HOG features) of each color channel of a color space. The results of these extracted features for each set of images are then stacked and normalized (using sklearn's `StandardScaler` method) and then split into training and testing datasets (using sklearn's `train_test_split` method).

I then explored different color spaces like HSV, HLS, RGB, YUV and YUV worked the best. Next was exploration of  different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I have built a classifier and tried to detect cars using these paramters in an image. The following parameters of HOG in YUV color space are the best set of paramters that worked well on the test images in detecting cars.

```
color space = 'YUV'
orientations=8
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)
```

#### Y image of car in YUV space:
[cary]:./out/cary.jpg
[hogy]:./out/hogy.jpg
![alt text][cary]

#### Hog visualization of Y image of car in YUV space:
![alt text][hogy]


#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have only used HOG features and not included color or spatial features to avoid very huge feature vectors. Extracting features and training the classifier is implemented in lines `127 through 179` in `project5.py`. I have divided the data(car and non car images) into training and testing datasets and then extracted features (HOG) using `extract fetures` function. I have then scaled and normalized the features to zero mean and unit variance and fed to Support vector machine classifier with appropriate labels(car or not car). The model learnt by the classifier was tested to predict labels on the testing dataset. I achieved an accuracy score of about 98% on the testing data set

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This is implemented in `lines 189-229` in `project5.py` where I have a function for sliding the window and searching for vehicles in this window. I have only searched for cars in the lower half of the image which is mainly the roads. The sliding window overlaps for 0.75 (both x and y).These values are chosen by experimenting different combinations and the ones which are fast and have obtained good detections have been chosen. The classifier checks if there is a car in the frame and all windows where a vehicle is predicted is returned. The cars which are nearer appear bigger and which are further appear smaller. So I have restricted the larger windows to lower part of the search area and I have also searched on three scales of window sizes small, medium and large in order to be able to able to detect cars in all positions in an image which are found to be best on the test images. The following are the window sizes in x and y directions I have used which also in `lines 189 to 229` in `project5.py`

```
75,75 
100,100
125,125
```
Here is the image containing the list of windows that are ususally searched for:
[allwindow]: ./out/allwindows.jpg
![alt text][allwindow]





#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features. The classifier predicted whether a given window has a car with some confidence. The windows that returned high confidence are selected to be robust to false positives and then I created a heatmap which is similar to probability of detecting a car in a window. If there are multiple windows in a region then the heatmap has high value and indicates higher probability of detecting a car which can be done by thresholding heatmap after trying different values for a threshold and the best threshold for which I could reasonably detect cars and eliminate false positives is 2. This is all implemented in lines `search_windows` in `lines 233 to 271`
The following image shows all the windows returned by the `search_windows` where the car is detected.

[allboxes]: ./out/all_boxes_car.jpg
![alt text][allboxes]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
False positive may occur when a window has no car and was detected to be a car. When this happens the heat may not be dense in that region and choosing a good threshold can eliminate the problem to a good extent. So I have thresholded the heatmap by a value (2, whcih was found by experimenting with different values on test images) This helped me to get rid of many of the False positives.
 
To combine the result of overlapping windows I have used connectedness property of pixels in a heat map to combine all the combine the overlapping windows using `scipy.ndimage.measurements.label()`. The following are the results obtained:
[heat_map]:./out/heat_map.jpg
[all_boxes]:./out/all_boxes_car.jpg
[finalimage]: ./out/final_car_boxed.jpg
#### Heat map
![alt text] [heat_map]
#### All the windows correspoding to heat map
![alt text][all_boxes]
#### Combined window using scipy.ndimage.measurementes.label() by using [8-connected](https://en.wikipedia.org/wiki/Pixel_connectivity#8-connected)
![alt text][finalimage]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I feel there should be a better heuristic for selcting parameters like the colorspace and hyper parameters of HOG  which were experimented by checking their performance on the test images. The pipeline is likely to fail in case of occlusions and also if there are there are False positives between true positives. To make it more robust Bag of words type of approaches could be used to handle occlusions.

