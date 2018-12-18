# **Traffic Sign Recognition** 

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

[image1]: /test_images/image1.png
[image2]: /test_images/image2.jpg
[image3]: /test_images/image3.jpg
[image4]: /test_images/image4.png
[image5]: /test_images/image5.jpg
[image6]: /test_images/image6.png
[Normalized_resized_image]: Normalized_resized_image.png
[bar_chart]: bar_chart.png

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I imported the dataset files and used the methods len() and shape() to calculate number of examples and the shape of the images.

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing no of training samples for each class.

![alt text][bar_chart]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I shuffled the examples using shuffle from the sklearn.utils library. X_train, y_train = shuffle(X_train, y_train)

I also tried augmenting the dataset and transforming the image but didn't result in a better prediction.

Finally, I normalized the image data to increase accuracy.

![alt text][Normalized_resized_image]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					                    | 
|:---------------------:|:-----------------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							                    | 
| Convolution 1     	| (5,5,3,6) filter, 1x1 stride, valid padding, outputs 28x28x6      |
| RELU 1				|												                    |
| Max pooling 1	      	| (2,2) filter, 2x2 stride, valid padding, outputs 14x14x6 		    |
| Convolution 2  	    | (5,5,6,16) filter, 1x1 stride, valid padding, outputs 10x10x16    |
| RELU 2				|												                    |
| Max pooling 2	      	| (2,2) filter, 2x2 stride, valid padding, outputs 5x5x16 			|
| Flatten				| Input = 5x5x16. Output = 400									    |
| Fully connected 1	    | Input = 400. Output = 120 				                        |
| RELU 3				|												                    |
| Fully connected 2		| Input = 120. Output = 84        									|
| RELU 4				| 									                                |
| Fully connected 3		| Input = 84. Output = 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used adam optimizer considered superior to stochastic gradient optimizer, 20 epochs, a batch size of 128 and a learning rate of 0.001.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.974

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My solution is based on well known LeNet architecture. This architecture is good and packages convolution, max pooling and fully connected layers that help is classification of images. I tried to replace max pooling with dropout regularization but that didn't help to improve the accuracy. I adopted steps such as normalization of images and hyperpararameter tuning such as using 20 epochs to achieve a validation accuracy of 0.967. Arriving at ideal normalization formula and number of epochs took some time and has been a iterative process. These are achieved in step 2 of the jupyter notebook. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]

The first, second, fourth and sixth images are relatively easy to classify as they are with perspective and are predicted well as per expectation. The third and fifth images are difficult to classify. The third image may be difficult as detection depends on sharp curve on right boundary in the image. The fifth image detection on the number inside the plate. As we can see, our model failed to predict well on these test images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were and the accuracy on these new predictions

Here are the results of the prediction:

| Image			                | Prediction	        					    | Accuracy                     |
|:-----------------------------:|:---------------------------------------------:| :---------------------------:|
| General caution      	        | General caution  								| 100%						   |
| No entry     			        | No entry 										| 100%                         |
| Road narrows on the right	    | Speed limit (70km/h)							| 100%                         |
| Roundabout mandatory	        | Roundabout mandatory					 		| 99.98%                       |
| Speed limit (20km/h)			| Speed limit (100km/h)      					| 100%                         |
| Road work		                | Road work      							    | 100%                         |


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.67%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image, the model is sure that this is a General caution sign (probability of 1.0), and the image does contain a General caution sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution   									| 
| 0.0     				| Pedestrians									|
| 0.0					| Traffic signals								|
| 0.0	      			| Wild animals crossing					 		|
| 0.0				    | Road narrows on the right      				|

For the second image, the model is sure that this is a No entry sign (probability of 1.0), and the image does contain a No entry sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry   									| 
| 0.0     				| Speed limit (20km/h)							|
| 0.0					| Speed limit (30km/h)							|
| 0.0	      			| Speed limit (50km/h)					 		|
| 0.0				    | Speed limit (60km/h)    						|

For the third image, the model is sure that this is a Speed limit (70km/h) sign (probability of 1.0), and the image does not contain a Speed limit (70km/h) sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (70km/h)  						| 
| 0.0     				| Speed limit (20km/h)							|
| 0.0					| Speed limit (30km/h)							|
| 0.0	      			| Speed limit (50km/h)					 		|
| 0.0				    | Speed limit (60km/h)     						|

For the fourth image, the model is sure that this is a Roundabout mandatory sign (probability of 0.9998), and the image does contain a Roundabout mandatory sign. The top five soft max probabilities are
Roundabout mandatory: 99.98%
Keep right: 0.02%
Children crossing: 0.00%
Right-of-way at the next intersection: 0.00%
Ahead only: 0.00%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9998         		| Roundabout mandatory  						| 
| 0.0002     			| Keep right 									|
| 0.0					| Children crossing								|
| 0.0	      			| Right-of-way at the next intersection		    |
| 0.0				    | Ahead only      							    |

For the fifth image, the model is sure that this is a Speed limit (100km/h) sign (probability of 1.0), and the image does not contain a Speed limit (100km/h) sign. The top five soft max probabilities are
Speed limit (100km/h): 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (100km/h) 						| 
| 0.0     				| Speed limit (20km/h)							|
| 0.0 					| Speed limit (30km/h)							|
| 0.0 	      			| Speed limit (50km/h)					 		|
| 0.0 				    | Speed limit (60km/h)    						|

For the sixth image, the model is sure that this is a Road work sign (probability of 1.0), and the image does contain a Road work sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work  									| 
| 0.0     				| Dangerous curve to the right					|
| 0.0					| Slippery road								    |
| 0.0	      			| No passing for vehicles over 3.5 metric tons	|
| 0.0				    | Right-of-way at the next intersection    		|
