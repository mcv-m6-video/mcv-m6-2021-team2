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

## __Week 1__
---------

On the first week the goal is to study the evaluation metrics needed for the system, in order also to be confortable with the data available. The metrics are:

* **Object Detection**:
 	* Mean Intersection over Union
    * Mean Average Precision

* **Optical Flow**:
    * Mean Square Error in Non-occluded areas
    * Percentage of Erroneous Pixels in Non-occluded areas

## __Week 2__
---------

__Task 1.1 Gaussian modelling__

1. Gaussian function to model each background pixel
    - First 25% of the test sequence to model background
    - Mean and variance of pixels

2. Second 75% to segment the foreground and evaluate

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1aV761QHabmQQitoXGatwK7zc4g2Yebu1JaH6rmS8xrg/edit#slide=id.gc86345e5f4_0_0)

__Task 1.2 Evaluation__

* Evaluate Task 1
    - mAP on detected connected components
    - Filter noise and group in objects
    - Over alpha threshold
    - Decide (and explain) if parked/static cars are considered

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1aV761QHabmQQitoXGatwK7zc4g2Yebu1JaH6rmS8xrg/edit#slide=id.gc86345e5f4_0_117)

__Task 2.1 Adaptative modelling__

* Adaptive modelling
    - First 25% frames for training
    - Second 75% left background adapts

* Best pair of values (𝛼, ⍴) to maximize mAP
* Two methods:
    - Obtain first the best 𝛼 for non-recursive, and later estimate ⍴ for the recursive cases
    - Optimize (𝛼, ⍴) together with grid search or random search (discuss which is best…).

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1aV761QHabmQQitoXGatwK7zc4g2Yebu1JaH6rmS8xrg/edit#slide=id.gc86345e5f4_0_165)

__Task 2.2 Comparison adaptative vs non__

Compare both the adaptive and non-adaptive version and evaluate them over mAP measures

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1aV761QHabmQQitoXGatwK7zc4g2Yebu1JaH6rmS8xrg/edit#slide=id.gc86345e5f4_0_221)

__Task 3 Comparison with state-of-the-art__

* Compare with state-of-the-art
    - P. KaewTraKulPong et.al. An improved adaptive background mixture model for real-time tracking with shadow detection. In Video-Based Surveillance Systems, 2002. Implementation: BackgroundSubtractorMOG (OpenCV)
    - Z. Zivkovic et.al. Efficient adaptive density estimation per image pixel for the task of background subtraction, Pattern Recognition Letters, 2005. Implementation: BackgroundSubtractorMOG2 (OpenCV)
    - L. Guo, et.al. Background subtraction using local svd binary pattern. CVPRW, 2016. Implementation: BackgroundSubtractorLSBP (OpenCV)
    - M. Braham et.al. Deep background subtraction with scene-specific convolutional neural networks. In International Conference on Systems, Signals and Image Processing, 2016. No implementation (https://github.com/SaoYan/bgsCNN similar?)

Evaluate to comment which method (single Gaussian programmed by you or state-of-the-art) performs better

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1aV761QHabmQQitoXGatwK7zc4g2Yebu1JaH6rmS8xrg/edit#slide=id.gc8748309c6_0_2)

## __Week 3__
---------

This week, we use the pre-trained models RetinaNet and Faster R-CNN on the AiCityChallenge video and evaluate their performance.

__Task 1.1 Off-the-shelf__

We used [Detectron2 Zoo Models](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md):
* Faster R-CNN - With R50 FPN Backbone
* RetinaNet - With R50 FPN Backbone

The Faster R-CNN and RetinaNet mAP50 are computed applying inference using a Detectron2
re-trained model using the [official detectron2 tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/10_nkRL2EFLIEEQEloHUi6XBRzcer_N9eBgKG--AvKks/edit#slide=id.gc9fea8d1a5_22_0)

__Task 1.2 Fine-tune to your data__

Fine tuning from your data will in general require two steps:

* Defining the new dataset
    - (eg. here for Faster R-CNN)
* Fine-tuning the last layer(s) from a pre-trained model
    - (eg. here for Faster R-CNN)

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/10_nkRL2EFLIEEQEloHUi6XBRzcer_N9eBgKG--AvKks/edit#slide=id.gc9fea8d1a5_22_15)

__Task 1.3 K-Fold Cross-Validation__

Try different data partitions on your sequence:

__Strategy A__ (same as week 2):
* First 25% frames for training
* Second 75% for test.

__Strategy B__:
* K-Fold cross-validation (use K=3).
* First 25% Train - last 75% Test (same as Strategy A).

__Strategy C__:
* K-Fold cross-validation (use K=3)
* Random 25% Train - rest for Test

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/10_nkRL2EFLIEEQEloHUi6XBRzcer_N9eBgKG--AvKks/edit#slide=id.gc9fea8d1a5_22_26)

__Task 2.1 Tracking by maximum overlap__

Basic algorithm (your task is to modify / improve it based on your experiments):

1. Assign a unique ID to each new detected object in frame N.
2. Assign the same ID to the detected object with the highest overlap (IoU) in frame N+1.
3. Return to 1.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/10_nkRL2EFLIEEQEloHUi6XBRzcer_N9eBgKG--AvKks/edit#slide=id.gc9fea8d1a5_1_0)

__Task 2.2 Tracking with a Kalman filter__

You may get inspiration from [this tutorial](https://www.amaiasalvador.com/) by Amaia Salvador to track the detections with a Kalman filter.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/10_nkRL2EFLIEEQEloHUi6XBRzcer_N9eBgKG--AvKks/edit#slide=id.gc9fea8d1a5_1_40)


## __Week 4__
-----

The objective of week 4 is to study different methods of optical flow, choose one and apply it in the video of the AICity challenge to track the cars.

__Task 1.1 Optical flow with block matching__

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

__Task 1.2 Off-the-shelf Optical Flow__

To compare the different Block Matching algorithms tried on the previous task, we have used some already implemented algorithms for optical flow estimation.

For these reason we have tried the following methods:
* [PyFlow](https://github.com/pathak22/pyflow)
* [Hornschunk](http://dspace.mit.edu/handle/1721.1/6337)
* [Lucas-Kanade](http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf)
* [Gunnar-Farneback](http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdfTwo-Frame)

To evaluate these algorithms we have used the MSEN and PEPN to compare them quantitatively and also plot the optical flow with HSV and vector plots to see their differences qualitatively.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gce876e9453_0_29)

__Task 2.1 Video Stabilization with Block Matching__

In this task, we want to stabilize a video using the optical flow estimation. To do so, we have used the best parameters obtained from the task 1.1 and also the best off-the-shelf method (considering MSEN/PEPN and time elapsed) which is the farneback.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gcecb7f29fd_1_19)

__Task 2.1 Off-the-shelf stabilization__

In this task we used off-the-shelf methods seen in task 1.2. We Used implementation from [Python video stabilization](https://github.com/AdamSpannbauer/python_video_stab).

Based on Optical Flow computation, described [here](http://nghiaho.com/?p=2093) which offers different implementations differing on keypoint extractor algorithm.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gcecb899401_4_0)

__Task 3.1 Object tracking with Optical Flow__

Finally, based on the best video stabilization method seen in previous tasks we apply it to perform tracking on the AICity challenge video. We compared the results obtained against the week 3.

Experiments and more detailed information can be found [here](https://docs.google.com/presentation/d/1hCgceBehTA5_3Vs2I9-sR5rmQzM5iTKulkw-dRkpIDQ/edit#slide=id.gcecb899401_7_3)

## __Week 5__
----
During this week we have tackled the problems of Multi Target Single Camera (MTSC) and Multi Target Multi Ccamera (MTMC)

The intention first is to recap and see which of our methods during all the weeks performs best against a baseline with the objective of tracking cars

Then the intention is to find an algorithm or method to be able to track cars in multiple cameras.
Our method is based on graph models to be able to track the same cars thanks to previously computed embeddings, in order to not depend on spatially neither temporal problems between the appearances from cars in the different cameras.

The **SLIDES** can be found [here](https://docs.google.com/presentation/d/1SKt3O-y2PFHsqUoh_Kva37HYnKhV8MT_PtB4mstIb-w/edit?usp=sharing)

Report link
