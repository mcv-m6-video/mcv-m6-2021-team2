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