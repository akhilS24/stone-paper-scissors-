# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:07:58 2020

@author: akku
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#initializing the classifier object
classifier= Sequential()

#creating convolutional layer
classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

#pooling the convolved layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding  second convolution layer
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening the layers
classifier.add(Flatten())

#making a full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation ='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=3, activation='softmax'))

#compiling 
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

#fitting the cnn to the images
train_data= ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)
train_generator = train_data.flow_from_directory(
        'train_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
test_generator = test_data.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

train_generator.class_indices


#train the model
classifier.fit(
        train_generator,
        steps_per_epoch=2520,
        epochs=5,
        validation_data=test_generator,
        validation_steps=372)



#loading the saved model
from keras.models import load_model
model=load_model('model2.h5')
model.summary()



#making a single prediction
import numpy as np
from keras.preprocessing import image

single_image=image.load_img('single_pred/test2.jpg', target_size=(64, 64))
single_image=image.img_to_array(single_image)
single_image=np.expand_dims(single_image, axis=0)
pred=model.predict(single_image)

x, y,z=round(pred[0][0]),round( pred[0][1]),round(pred[0][2])
if((x, y, z)==(1, 0, 0)):
    print('paper')
elif((x, y, z)==(0, 1, 0)):
    print('rock')
elif((x, y, z)==(0, 0, 1)):
    print('scissors')
    

    
