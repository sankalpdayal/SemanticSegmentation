## Deep Learning Project ##

In this project, I trained a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques applied here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./images/SemanticSegmentation.JPG
![Semantic Segmentation][image_0] 

## Fully Convolutional Networks (FCN) ##
A Fully Convolutional Networks (FCN) Architecture was implemented to perform the task of semantic segmentation. This kind of actichtecutre  This FCN architecture has proven to have good results for segmentation tasks. 
This comprises of 3 blocks; Encoders, 1x1 Convolutional Layer and Decoders. This architecture allows first downsampling of the image to extract important information and then upsamples them to reach same size as input image. Since
most spatial information is lost, the last layers receive information from previous layer but also from initial layers called skip connections. These skip connections are critical for semantic segmentation tasks as 
it allows using information from multiple image sizes and layers of the architecture there by retaining more information.

### 1. Encoders
Encoders comprises of 1 or more layers that as imilar to regular convolutions as in CNNs. These layers try to learn like regular CNNs features of the objects in the image used for classification. To detect object, these layers for example can learn shape, colors, edges etc.
These encoders reduce the dimension of the input image but at an expense of losing spatial information. These layers are separable convolution layers.

### 2. 1x1 Convolution Layer
The 1x1 convolution layer is a regular convolution, with a kernel and stride of 1. The importance of this layer is that it allows retaining of spatial information from the encoder. 

### 3. Decoders
Decoders perform the task of upsampling of input information. The can be either composed of transposed convolution layers or bilinear upsampling layers. In transposed convolutions the process 
involves multiplying each pixel of your input with a kernel or filter. If this filter was of size 5x5, the output of this operation will be a weighted kernel of size 5x5. This weighted kernel then defines your output layer. Bilinear upsampling is similar to 'Max Pooling' and uses the weighted average of the four nearest known pixels from the given pixel, estimating the new pixel intensity value. 

The decoder block receieves a concatenated layer of output from previous layer and output from encoder layer. This is the skip connection. It is done by calculating the  bilinear upsample of the smaller input layer 
and concatenated with the larger input layer coming from encoder. This concatenated layer is given to a separable convolution layer. This whole setup is called a decoder. A single FCN can have multiple decoders.


## Network Architecture ##
My final network has 3 encoders, single 1x1 Convolution Layer and 3 decoders. The implementation of the model is in the section `Model` in the notebook `model_training.ipynb`. The design showing layers sizes, connections and skip connections in the following diagram.

[image_1]: ./images/Architecture.JPG
![Architecture][image_1] 

## Choice of Design and Hyperparameters ##
I tried different design options and different hyper parameters. I tried 3 and 4 encoder and decoder pairs. I tried different learning rates, batch sizes, number of epochs and steps per epochs. I used the curve for training and validation error with respect to epochs to see how close the errors are and how the model is evolving. I made sure that error are close and converged.
Initially I kept 3 ecnoders and 3 decoders. I tried to make the training and validation error very close. Here I tried different filter size for 1x1 convolution layer. Once they came close, I reducded the learning rate and tried different batch sizes, number of epochs and steps per epochs.
I also tested architecture with 4 ecnoders and 4 decoders.  Following table shows the summary of experiments that I performed. Please note for the encoders I have given filters and strides in format <filters>-<strides>. Each layer is separated by a comma. 
For others I have given only filters. Trial 5 is the one which I have considered final and is checked in the github repository.

|Trial| Encoders| 1x1 Convolution|Decoders|Learning Rate| Batch Size  | Num Epochs | Steps Per Epoch | Validation Steps |
|-----|---------|----------------| -------|-------------|-------------|------------|-----------------|------------------|
|1|32-2,64-2,128-2| 8| 128,64,32 |0.001|64|25|65|50|
|2|32-2,64-2,128-2| 8| 128,64,32 |0.001|26|30|150|50|
|3|32-2,64-2,128-2|128| 128,64,32 |0.001|32|20|150|50|
|4|32-2,64-2,128-2|128| 128,64,32 |0.001|32|40|150|50|
|5|32-2,64-2,128-2|128| 128,64,32 |0.0001|32|40|150|50|
|6|32-2,64-2,128-2,256-2|256| 256,128,64,32 |0.0001|32|80|150|50|

Following plots show the training and validation errors for different trials.

[image_2]: ./images/TrialPlots.JPG
![Trial Plots][image_2] 

The training was performed on AWS EC2 p2.xlarge instance with workers as 2.

## Results ## 
The weights for the model can be found in file [/data/weights/model_weights](./data/weights/model_weights). When these weights and the model chosen is used I can following 

|Training Loss| 0.0122|
|Validation Loss| 0.0260|
|Final IoU| 0.5702|
|Final Score| 0.4230|

Once the model had achieved the desired final score, I tested the model in the simulation to allow the quad to find and then follow the target. First the quad searched for the target and once it found it followed it well. Following screenshots show different times when the quad was following the hero.

[image_3]: "./images/Screen Shot 2018-05-25 at 1.49.52 PM.png"
![Following 1][image_3] 

[image_4]: "./images/Screen Shot 2018-05-25 at 1.50.21 PM.png"
![Following 2][image_4]

[image_5]: "./images/Screen Shot 2018-05-25 at 1.51.05 PM.png"
![Following 3][image_5]  


## Discussion ##
This model is trained for a single person right now and all training is done in simulation. Important thing to note here is that the hero was always wearing different colored clothing and hence this could be possible that 
the model has learning color as a strong feature and would fail if the target changes clothers which in real life could be normal use case. Hence the training dataset needs to be extended for the same target with multiple cloth 
combinations. This will allow model to learn more information like face details, height, body composition etc. The model can likewise be trained for other objects like dogs, cat, car etc. if trained for enough number of images. But as mentioned earlier, 
the current model could have just learn colors as important features, for dogs, cats and cars, this can really be confusing as many breeds of animals and many models of cars can have same color. Hence enough diversity has to be given 
in data set to allow it to learn specific features like body composition of target dog or cat and license plate for the car. One thought comes is to use instead of just an image but a series of images to allow time based features as well. For human following
robot gait can become a strong feature here.

## Future Enhancements ##  
I would definitely try to increase dataset with images when target is far because current model is not able to identify target well when it is far. Also I would try playing with the more layers to allow the network to identify the target when it is far.
  
