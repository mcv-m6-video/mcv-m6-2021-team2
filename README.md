# Video Surveillance for Road Traffic Monitoring

The goal of the project is to provide a robust system for road traffic monitoring using computer vision techniques for video analysis. 

## Team 2
## Contributors
* [Ruben Bagan](https://github.com/rbagan)
* [Joan Fontanals](https://github.com/JoanFM)
* [Vernon Stanley](https://github.com/drkztan)
* [Pablo Domingo](https://github.com/paudom)

## Reproducibility

We have developed the code in `src/` and then for each week we have jupyter notebooks to where the final results will be exposed.

## Week 1

On the first week the goal is to study the evaluation metrics needed for the system, in order also to be confortable with the data available. The metrics are:

* **Object Detection**:
 	* Mean Intersection over Union
    * Mean Average Precision

* **Optical Flow**:
    * Mean Square Error in Non-occluded areas
    * Percentage of Erroneous Pixels in Non-occluded areas

## Week 2

The goal for this week is to model the background of a video sequence in order to estimate the foreground objects.
To do so, different tasks are developed:

* Estimate the background using a single gaussian per pixel (Non-Adaptative)
* Estimate the background with adaptative strategy
* Compare both approaches and also the State of the Art
* Use color within the background estimation

## Week 3 

This week, we use the pre-trained models RetinaNet and Faster R-CNN on the AiCityChallenge video and evaluate their performance.

Tasks:
* Use pre-trained RetinaNet/Faster R-CNN to detect cars in the image sequence.      🆗
* Train the previous models using our image sequence while fine-tuning the models.  🆗
* Use different validation strategies to evaluate the performance of the models.    🆗
* Track distinct, different objects in the sequence, assigning them a unique ID.    🆗
* Use different tracking methods like Kalman filters and maximum overlap.           🆗
* Evaluate the performance of the object trackers using IDF1 score.                 🆗

## Week 4

This week we estimate the optical flow given a previous and a current frame.

The estimation is done using Block Matching with different algorithms:
* Exhaustive Search
* Three Step Search

We also try different off-the-shelf methods to estimate the optical flow
* PyFlow
* Lucas Kanade
* Horn Schunk
* Farneback

Using the estimation we try to stabilize a video and also observe if by using optical flow we can improve the traking done in teh previous week (week3)

The task where:
* Use different parameteres of Block Matching and find the best ones to estimate Optical Flow  🆗
* Try Off-The-Shelf optical flow and comapre between them and also between Block Matching 🆗
* Use the optical flow to stabilize a video with a lot of motion 🆗
* Try different algorithms to stabilize the same video 🆗
* Use the estimation of optical flow in order to observe if we improve the tracking 🆗
