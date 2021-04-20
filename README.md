# Team 2 - Video Surveillance for Road Traffic Monitoring project

The goal of this project is to learn the basic concepts and techniques related to video sequences processing, mainly for surveillance applications. We will focus on video sequences from outdoor scenarios, with the application of traffic monitoring in mind.

The main techniques of video processing will be applied in the context of video surveillance:
* Background modelling
* Moving object segmentation
* Motion estimation and compensation
* Video object tracking

In a first stage, moving object segmentation will be tackled considering scenarios with static camera. Afterwards, camera motion will be considered. Tracking of multiple moving objects will be performed both in single-camera and multi-camera scenarios.

These tracking results will provide high level information that will be analysed for traffic monitoring (measuring speed and density of cars, detecting anomalies in traffic, etc.).

The learning objectives of the project will be the use of pixel based statistical models (such as mixture of gaussians) for modelling a scene background, the use of semantic segmentation architectures for moving object detection, the development of optical flow estimation methods for camera motion compensation and the analysis of deep learning architectures for single/multi camera object tracking.

## Contributors
* [Ruben Bagan](https://github.com/rbagan)
* [Joan Fontanals](https://github.com/JoanFM)
* [Vernon Stanley](https://github.com/drkztan)
* [Pablo Domingo](https://github.com/paudom)

## Install

In order to install the project, follow the next steps.

```
git clone https://github.com/mcv-m6-video/mcv-m6-2021-team2.git
cd mcv-m6-2021-team2
pip install -r requirements.txt
```

## Run

You can run the task of each week using

```
python weekX_task_Y_Z.py
```

Where `X` is the week number, `Y` the task and `Z` sub task.

## Summary of the different weeks

### Week 1

On the first week the goal is to study the evaluation metrics needed for the system, in order also to be confortable with the data available. The metrics are:

* **Object Detection**:
 	* Mean Intersection over Union
    * Mean Average Precision

* **Optical Flow**:
    * Mean Square Error in Non-occluded areas
    * Percentage of Erroneous Pixels in Non-occluded areas

### Week 2

The goal for this week is to model the background of a video sequence in order to estimate the foreground objects.
To do so, different tasks are developed:

* Estimate the background using a single gaussian per pixel (Non-Adaptative)
* Estimate the background with adaptative strategy
* Compare both approaches and also the State of the Art
* Use color within the background estimation

### Week 3

This week, we use the pre-trained models RetinaNet and Faster R-CNN on the AiCityChallenge video and evaluate their performance.

Tasks:
* Use pre-trained RetinaNet/Faster R-CNN to detect cars in the image sequence.
* Train the previous models using our image sequence while fine-tuning the models.
* Use different validation strategies to evaluate the performance of the models.
* Track distinct, different objects in the sequence, assigning them a unique ID.
* Use different tracking methods like Kalman filters and maximum overlap.
* Evaluate the performance of the object trackers using IDF1 score.

## __Week 4__
-----

The objective of week 4 is to study different methods of optical flow, choose one and apply it in the video of the AICity challenge to track the cars.

__Task 1.1 Optical flow with block matching__ (`Week4_task1_1.py`)

* Implement a block matching solution for optical flow estimation.
* Configurations and parameters to explore:
    - Forward or Backward compensation
    - Area of Search
    - Size of the blocks
    - Error function
    - Step size

The algorithms implemented are (both implemented from scratch):
* Exhaustive Search
* Three Step Search

For the three step search we used the [following paper](https://www.researchgate.net/publication/290150215_A_New_Survey_on_Block_Matching_Algorithms_in_Video_Coding) written by L.C.Manikandan and Dr. R.K.Selvakumar. To learn the common block matching algorithms and how to implement them.

We decided to implement an extra algorithm (Three step search) based on the following paper [A Comparison of Different Block Matching Algorithms for Motion Estimation](https://www.sciencedirect.com/science/article/pii/S2212017313003356) written by Razali Yaakob, Alihossein Aryanfar, Alfian Abdul Halin and Nasir Sulaiman.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gce876e9453_0_4)

__Task 1.2 Off-the-shelf Optical Flow__ (`week4_task1_2.py`)

To compare the different Block Matching algorithms tried on the previous task, we have used some already implemented algorithms for optical flow estimation.

For these reason we have tried the following methods:
* [PyFlow](https://github.com/pathak22/pyflow)
* [Hornschunk](http://dspace.mit.edu/handle/1721.1/6337)
* [Lucas-Kanade](http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf)
* [Gunnar-Farneback](http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdfTwo-Frame)

To evaluate these algorithms we have used the MSEN and PEPN to compare them quantitatively and also plot the optical flow with HSV and vector plots to see their differences qualitatively.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gce876e9453_0_29)

__Task 2.1 Video Stabilization with Block Matching__ (`week4_task2_1.py`)

In this task, we want to stabilize a video using the optical flow estimation. To do so, we have used the best parameters obtained from the task 1.1 and also the best off-the-shelf method (considering MSEN/PEPN and time elapsed) which is the farneback.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gcecb7f29fd_1_19)

__Task 2.1 Off-the-shelf stabilization__ (`week4_task2_2.py`)

In this task we used off-the-shelf methods seen in task 1.2. We Used implementation from [Python video stabilization](https://github.com/AdamSpannbauer/python_video_stab).

Based on Optical Flow computation, described [here](http://nghiaho.com/?p=2093) which offers different implementations differing on keypoint extractor algorithm.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gcecb899401_4_0)

__Task 3.1 Object tracking with Optical Flow__ (`week4_task3_1.py`)

Finally, based on the best video stabilization method seen in previous tasks we apply it to perform tracking on the AICity challenge video. We compared the results obtained against the week 3.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gcecb899401_7_3)

## __Week 5__
----

Slides link

Report link