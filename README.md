                          DEEP LEARNING ACTIVITY 1 
			 IMAGE CLASSIFICATION USING FRUIT DATA


The dataset for the image classification and machine deployment activity was taken from Kaggle. 

The link for the dataset is given below:https://www.kaggle.com/code/harinuu/fruit-classification-cnn/log

	DATA COLLECTION

The dataset for this activity consists of hundreds of images of various fruits of 33 different categories. As the activity focuses heavily on image classification,
the choice of selection for the dataset was decided accordingly.

The dataset has 33 categories of fruits from which we have selected two categories for the image classification activity. 

The two fruits selected are 'Apple Braeburn' and 'Banana.' There are 491 images of an apple and 489 images of a banana available in the new version of the dataset that was created to meet our needs.

Using this dataset we want to train the model to accurately predict the images according to the image uploaded in the Gradio interface.

	PREPROCESSING AND MODEL BUILDING

After the selection of the dataset that we are using for image classification, we can then proceed with the next step; that is the preprocessing of the data and model training.


We now import the necessary libraries and modeules.
		 
	import numpy as np

	import os

	import matplotlib.pyplot as plt

	import tensorflow as tf

	import PIL

	import pathlib

	from tensorflow import keras  

	from tensorflow.keras import layers, models


These lines import the necessary libraries and modules for the activity. Numpy is used for numerical operations, and the OS module in Python provides functions for interacting with the operating system. TensorFlow is a library for multiple machine learning tasks, while Keras is a high-level neural network library that runs on top of TensorFlow. Both provide high-level APIs(Application Programming Interface) used for easily building and training models. Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. 


	image_size = 100, 100

	batch_size = 32


img_size: This represents the height and width dimensions of the input images in pixels. In this case, both dimensions are set to 100 pixels, which means that the model expects input images to be 100 pixels in height and 100 pixels in width.

batch_size: This represents the number of data samples that will be processed in each iteration (or batch) during training or inference. A batch size of 32 means that 32 images (or data samples) will be processed in each training iteration.

Here we are mounting Google Drive in a Google Colab environment.Mounting Google Drive in Colab allows access to work with files stored in your Google Drive directly within your Colab notebooks.
 			 
	from google.colab import drive

	drive.mount('/content/drive')
 

We are creating training, validation, and test datasets using the image_dataset_from_directory function in TensorFlow. These datasets will be useful for training and evaluating the models on image data located in the specified directory on the Google Drive.

 	train_dataset = tf.keras.preprocessing.image_dataset_from_directory(

 	'/content/drive/MyDrive/dldata/dlimg',
		
 	shuffle=True, batch_size=batch_size, image_size=img_size,
		 
 	validation_split=0.2, subset='training', seed=42)

 	)


 	valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(

 	'/content/drive/MyDrive/dldata/dlimg',
		
 	shuffle=True, batch_size=batch_size, image_size=img_size,
		
 	validation_split=0.2, subset='validation', seed=42

	)
 
 	test_dataset = tf.keras.preprocessing.image_dataset_from_directory(

 	'/content/drive/MyDrive/dldata/dlimg',
		
 	shuffle=True, batch_size=batch_size, image_size=img_size
		
	)

With these datasets, you can now proceed to train and evaluate our CNN model.

	class_names = train_dataset.class_names
 
	print(class_names)

Here, we are printing the class names that are 'Apple Braeburn' and 'Banana' from the training dataset.

	plt.figure(figsize = (15 ,15))
 
	for images, labels in train_dataset.take(8):#take() returns images in given positional indices along the axis
 
    	for i in range(14):
		 
        	ax = plt.subplot(4, 4, i + 1)
				 
        	plt.imshow(images[i].numpy().astype("uint8")) #imshow() creates an image from a 2d numpy array
				 
        	# unit8 is an unsigned integer ranging from 0 to +255
				 
        	plt.title(class_names[labels[i]])
				 
        	plt.axis("off")

The above few lines of code are intended to display a grid of images from the train_dataset, where each row contains 4 images, and there

are a total of 8 rows (32 images in total). It also labels each image with its corresponding class name. This is done to visualize a

portion of the training dataset.

	num_classes=2

num_classes is equal to 2 implies that this is a binary classification problem where there are two classes

or categories.The two classes are 'Apple Braeburn' and 'Banana'.

	def cnnmodel(lr=0.001):
 
    	model = models.Sequential([
		 
       	  layers.Rescaling(1./255),
					
        	layers.Conv2D(32, 3, activation='relu'),
				 
        	layers.BatchNormalization(),
				 
        	layers.MaxPooling2D((2, 2)),
				 
        	layers.Conv2D(64, 3, activation='relu'),
				 
        	layers.BatchNormalization(),
				 
        	layers.MaxPooling2D((2, 2)),
				 
        	layers.Conv2D(128, 3, activation='relu'),
				 
        	layers.BatchNormalization(),
				 
        	layers.MaxPooling2D((2, 2)),
				 
        	layers.Flatten(),
				 
        	layers.Dense(128, activation='relu'),
				 
        	layers.Dropout(0.5),  # Adjust dropout rate
				 
        	layers.Dense(num_classes)
    	])

    	model.compile(optimizer=keras.optimizers.Adam(lr),
		 
                  	loss='sparse_categorical_crossentropy',
									 
                  	metrics=['accuracy'])

    	return model

The cnnmodel function defines a Convolutional Neural Network (CNN) model for image classification. 

1. layers.Rescaling(1./255) scales the pixel values of the input images to the range [0,1]. This is important because neural networks tend

   to perform better when the input data is in a normalized range.

3. layers.Conv2D(32, 3, activation='relu') is a convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation. It's the 

   first convolutional layer in the model, responsible for capturing low-level features in the input images.

3. layers.BatchNormalization() is used to normalize the activations of the previous layer. It helps in stabilizing and accelerating

   training.

5. layers.MaxPooling2D((2, 2)) is a max-pooling layer with a 2x2 pooling window. Max-pooling is used to downsample the spatial dimensions

   of the feature maps.

6. layers.Conv2D(64, 3, activation='relu') is another convolutional layer with 64 filters and ReLU activation. It increases the complexity

   of the features captured.

6. layers.BatchNormalization() is applied again after the second convolutional layer.

7. layers.MaxPooling2D((2, 2)) is another max-pooling layer to downsample.

8. layers.Conv2D(128, 3, activation='relu') is the third convolutional layer with 128 filters and ReLU activation. This layer further 

   increases feature complexity.

9. layers.MaxPooling2D((2, 2)) is the final max-pooling layer.

10. layers.Flatten() here flattens the output from the previous layers into a 1D vector to feed into the fully connected layers.

11. layers.Dense(128, activation='relu') is a fully connected (dense) layer with 128 units and ReLU activation. This layer captures higher-

    level features.

12. layers.Dropout(0.5) is the dropout layer with a dropout rate of 0.5. Dropout helps in preventing overfitting by randomly dropping a 

    fraction of neurons during training.

13. layers.Dense(num_classes) is the output layer with as many units as there are classes in your classification problem. It uses a

    softmax activation function to produce class probabilities.

15. model.compile() compiles the model with the specified optimizer, loss function, and metrics for training.

This is an important starting point for an image classification mode and depending on the specific dataset and problem, there  may be a 

need to fine-tune hyperparameters, adjust the model architecture, or implement other techniques to improve performance. 

	model = cnnmodel()
 
	epochs=10
 
	history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)


The above code is for training a convolutional neural network (CNN) model using TensorFlow/Keras. 

1. model = cnnmodel() creates an instance of the CNN model using the cnnmodel function that is defined earlier. This function constructs 

  and compiles the model architecture based on the specified configuration. Essentially, it initializes the neural network model.

2. epochs = 10 specifies the number of training epochs. An epoch is one complete pass through the entire training dataset. In this case,

   it is set to 10, which means that the model will be trained for 10 full passes through the training data.

3. history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)is the training step of your model. 

 	The training dataset is provided as the input data for training. It typically consists of batches of images and their corresponding

 	labels. The fit method will iterate through this dataset for the specified number of epochs to train the model.

 	validation_data=valid_dataset argument is used to provide a validation dataset that the model will use to monitor its performance
	
 	during training.
	
 	The model's performance on the validation dataset is computed after each epoch and can help understand how well the model is

 	generalizing to data it hasn't seen during training.

5. history is a variable that stores the training history, which includes metrics (such as loss and accuracy) recorded during each epoch.

  It can be used later for plotting training curves and evaluating model performance.


	model.evaluate(test_dataset)

This is used to evaluate the performance of the trained machine-learning model on a test dataset. 

	# Printing the Loss and Accuracy
 
	(loss, accuracy) = model.evaluate(valid_dataset)
 
	print(loss)
 
	print(accuracy

	model.save('model.h9')

After evaluating the trained model on the validation dataset, it retrieves the loss and accuracy metrics and saves the model.

The .save() is used to save the entire model, including its architecture, trained weights, and optimizer configuration so that the model 

can later be loaded and reused without having to retrain it.

	image_batch, label_batch = test_dataset.as_numpy_iterator().next()
 
	predictions = model.predict_on_batch(image_batch)

	predictions = np.argmax(predictions, axis=-1)

	print('Predictions:\n', predictions)
 
	print('Labels:\n', label_batch)

	plt.figure(figsize=(10, 10))
 
	for i in range(9):
 
  	ax = plt.subplot(3, 3, i + 1)
	 
  	plt.imshow(image_batch[i].astype("uint8"))
	 
  	plt.title(class_names[predictions[i]])
	 
  	plt.axis("off")
 
The code is used to make predictions using the trained model on a batch of test images and visualize the results.

The result is a visual representation of the model's predictions for a batch of test images, allowing see how well the model is performing on this specific subset of data.




 



