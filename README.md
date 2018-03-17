# Project: Build Lane Line Detection Program

This repo contains codes and data needed to complete the Project 1 of Udacity Self-Driving Car Nanodegree.
The program is designed to detect road lanes in images or videos.

## Environment

This repo includes a yml file to create the environment for this project and you can run the following command:

```
conda env create -f environment.yml
```

## Overview

Given a image or video of traffic road, the program can automatically detect lane lines as shown below.

<p float="left">
  <img src="https://github.com/lipeng2/CarND-LaneLines-P1/blob/master/test_images/solidWhiteRight.jpg" width="300" />
  <img src="https://github.com/lipeng2/CarND-LaneLines-P1/blob/master/test_images_output/output_solidWhiteRight.jpg" width="300" /> 
</p>

The codes are implemented in [P1.ipynb](https://github.com/lipeng2/CarND-LaneLines-P1/blob/master/P1.ipynb). 

The pipeline consisted of 6 main steps as followed:

  1. Change color channel of given image into grayscale using **cv2.cvtColor** method.
  2. Apply a Gaussian blur to the the image using **cv2.GaussianBlur** method.
  3. Apply Canny transformation to obtain edges on the image using **cv2.Canny** method.
  4. Crop the desire region of the image using **Region_of_interest** method.
  5. Deploy Hough transformation to find the lines on the cropped image using **cv2.HoughLinesP** method.
  6. Merge the original image with the result image form above using **weightred_img** method

The pipeline is tested using iamges in the [test_images](https://github.com/lipeng2/CarND-LaneLines-P1/tree/master/test_images), and results from 4 of the first 5 steps are saved in [gray](https://github.com/lipeng2/CarND-LaneLines-P1/tree/master/test_images/gray), [blur](https://github.com/lipeng2/CarND-LaneLines-P1/tree/master/test_images/blur), [canny](https://github.com/lipeng2/CarND-LaneLines-P1/tree/master/test_images/canny), and [hough_lines](https://github.com/lipeng2/CarND-LaneLines-P1/tree/master/test_images/hough_lines). Lastly, the final results of merged images are saved in [test_images_output](https://github.com/lipeng2/CarND-LaneLines-P1/tree/master/test_images_output). Additionally, the pipeline is also tested using [videos](http://localhost:8888/tree/CarND-LaneLines-P1/test_videos), and the results are stored in [test_videos_output](http://localhost:8888/tree/CarND-LaneLines-P1/test_videos_output)

## Potential shortcomings with current pipeline

One potential shortcoming is that the pipeline is not robust enough to perform well on lane curves with high degree of radian, which is illustrated with challenge output video in [test_videos_output](http://localhost:8888/tree/CarND-LaneLines-P1/test_videos_output) 

## Possible improvement

One possible improvement is to draw a smoother curve using **cv2.ployfit** and **cv2.ploylines** methods instead of **cv2.lines**. 
