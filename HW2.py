#!/usr/bin/env python3
#Jason Millette
#HW2 Dogs vs Cats
# 9/28/2019

#Importing libraries
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Needed for cuDNN version not sure why########################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
##############################################################

#Local directories for images
base_dir = '/home/jmoney420/Documents/ECE498/HW2/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

#Directory with cat pictures for training
train_cats_dir = os.path.join(train_dir, 'cats')

#Directory with dog pictures for training
train_dogs_dir = os.path.join(train_dir, 'dogs')

#Directory with cat pictures for validation
validation_cats_dir = os.path.join(validation_dir, 'cats')

#Directory with dog pictures for validation
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#Printing file names
train_cat_fnames = os.listdir(train_cats_dir)
print (train_cat_fnames[:10])

train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()
print (train_dog_fnames[:10])

validation_cat_fnames = os.listdir(validation_cats_dir)
validation_dog_fnames = os.listdir(validation_dogs_dir)

#prints number of images
print ('total training cat images:', len(os.listdir(train_cats_dir)))
print ('total training dog images:', len(os.listdir(train_dogs_dir)))
print ('total validation cat images:', len(os.listdir(validation_cats_dir)))
print ('total validation dog images:', len(os.listdir(validation_dogs_dir)))

#%matplotlib inline


# Parameters for our graph; we'll output images in a 3x4 configuration
nrows = 4
ncols = 3

# Index for iterating over images
pic_index = random.randrange(0, (len(os.listdir(validation_cats_dir)) - 4)) #Credits to Spencer Goulette

# Set up matplotlib fig, and size it to fit 3x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
pic_index += 3 
next_train_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[pic_index*2-6:pic_index*2-3]]
next_train_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[pic_index*2-6:pic_index*2-3]]
next_validation_cat_pix= [os.path.join(validation_cats_dir, fname) 
                for fname in validation_cat_fnames[pic_index-3:pic_index]]
next_validation_dog_pix = [os.path.join(validation_dogs_dir, fname) 
                for fname in validation_dog_fnames[pic_index-3:pic_index]]

for i, img_path in enumerate(next_train_cat_pix+next_train_dog_pix+next_validation_cat_pix+next_validation_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i+1)
  sp.axis('On') # Don't show axes (or gridlines)
  ax = plt.gca()        #Credits to Spencer Goulette
  ax.axes.get_yaxis().set_visible(False)
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.set_xlabel(img_path[int(img_path.rfind('/')) + 1:])

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()  #Display plot with all 12 images





################### Assignment #2    ###########################

#Assigning weights
local_weights_file = '/home/jmoney420/Documents/ECE498/HW2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)

#make the model non trainable
for layer in pre_trained_model.layers:
  layer.trainable = False

#sets a 7x7 feature map
last_layer = pre_trained_model.get_layer('mixed7')
print ('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

#applies fully connected classifier to last_output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])
#Filenames
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

#Train model for 2 epochs on all 1000 test images
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=2,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)


##################  Assignment 3  ######################################################

#Confusion matrix
validation_generator.reset()
Y_pred = model.predict_generator(validation_generator, 50)  #50*match size = total images
y_pred = np.argmax(Y_pred, axis=1)
y_truth = validation_generator.classes[validation_generator.index_array]
print('Confusion Matrix')
cm = confusion_matrix(y_truth, y_pred)
print(cm)

#classification report
print(classification_report(y_truth, y_pred))
