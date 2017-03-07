#**Traffic Sign Recognition** 

##Project Writeup

###This writeup was based off the template provied

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_for_writeup/bar_chart_of_sign_count.png "Visualization"
[image2]: ./images_for_writeup/image1.png "As Loaded"
[image3]: ./images_for_writeup/image1_normal.png "Normalized"
[image4]: ./new_images/1.png "Traffic Sign 1"
[image5]: ./new_images/2.png "Traffic Sign 2"
[image6]: ./new_images/3.png "Traffic Sign 3"
[image7]: ./new_images/5.png "Traffic Sign 4"
[image8]: ./new_images/6.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/minakian/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###0. Loading the Data
The data set provided did not include a validation set. I took 20% of the test set and, using the train_test_split() function, derived a validation set.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library's shape instruction to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? -> x_train_shape[0] or 31367
* The size of test set is ? -> x_test_shape[0] or 12630
* The shape of a traffic sign image is ? -> [x_train_shape[1], x_train_shape[2]]
* The number of unique classes/labels in the data set is ? -> len(np.unique(y_train)) or 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed per sign type.

![alt text][image1]

I found another visualization by ottonello [their code](https://github.com/ottonello/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb), that nicely arranges and labels 12 images as well as providing a count of the signs with the labels of each. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 6th code cell of the IPython notebook.

As a first step, the only manipulation of the data I performed was to shuffle the images. After completing the training and validation, I wanted to see if I could increase the accuracy of the system with some kind of manipulation.

The first manipulation I attempted was to normalize the data between 0.1 and 0.9. This standardized the image formatting and eliminates any extremes in the image. This gave a marked validation improvement, from 93% to 98%

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]     ![alt text][image3]

I tried a few other steps, such as grayscaling, but this did not have a significant impact on the output.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets were described earlier in section 0..  

My final training set had 31367 images. My validation set and test set had ~7842 and 12630 images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 8th cell of the ipython notebook. I followed the LeNet architecture with a minor modification.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten				| Output 400									|
| Fully connected		| Input = 400 Output = 120 						|
| Relu  				| Activation									|
| Fully connected		| Input = 120 Output = 84 w/50% dropout			|
| Relu  				| Activation									|
| Fully connected		| Input = 84 Output = 43						|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 9th/10th cell of the ipython notebook. 

To train the model, I used the same code as described in the LeNet example which utilized an AdamOptimizer. I stuck with the default 128 batch size but changed from 10 epochs to 25. This could have been reduced back to 19/20, as validation accuracy stopped increasing after 19.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 10th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 98.4% 
* test set accuracy of 93.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    LeNet - We were provided a solid example in lecture.
* What were some problems with the initial architecture?
    Initial validation came in under 93%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    The only change to the architecture was to add dropout to the 2nd fully connected layer. This decision was made as a matter of trial, but worked well enough to improve the model.
* Which parameters were tuned? How were they adjusted and why?
Epochs were increased from 10 to 25. When the training was complete validation was still making significant increases and an increase in epochs allowed me to acheive a higher validation accuracy.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These were pretty clear images and the model had no trouble identifying them.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Wild Animals Crossing	| Wild Animals Crossing							|
| Yield					| Yield											|
| Road Work Ahead		| Road Work Ahead				 				|
| Stop					| Stop											|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93% accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is very sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry   									| 
| 1.69e-13 				| Double Curve 									|
| 5.9e-16				| Slippery Road									|
| 3.1e-21				| Bicycles Crossing				 				|
| 6.9e-23				| Beware of Ice/Snow							|


The data for the other images can be seein in the reults. 