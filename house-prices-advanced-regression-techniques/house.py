
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import keras


# In[7]:

# List our data sets
from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))

# Loading data with Pandas
train = pd.read_csv('data/train.csv')
train_images = train.ix[:,1:].values.astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

test_images = pd.read_csv('data/test.csv').values.astype('float32')

print(train_images.shape, train_labels.shape, test_images.shape)


# In[3]:

# Show samples from training data
show_images = train_images.reshape(train_images.shape[0], 28, 28)
n = 3
row = 3
begin = 42
for i in range(begin,begin+n):
    plt.subplot(n//row, row, i%row+1)
    plt.imshow(show_images[i], cmap=plt.get_cmap('gray'))
    plt.title(train_labels[i])


# In[8]:

# Normalize pixel values from [0, 255] to [0, 1]
train_images = train_images / 255
test_images = test_images / 255

# Convert labels from [0, 9] to one-hot representation.
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

from sklearn.model_selection import train_test_split
train_images, cv_images, train_labels, cv_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=0)

print(train_labels[0])
print(train_images.shape, train_labels.shape, cv_images.shape, cv_labels.shape)


# In[9]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D


# In[10]:

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[11]:

from keras.optimizers import RMSprop
from keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[12]:

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 20)


# In[13]:

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto'),
            ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0)]


# In[18]:

try:
    model.load_weights('mnist.h5')
except: 
    pass

hist = model.fit_generator(datagen.flow(train_images, train_labels, batch_size = 64),
                           steps_per_epoch = 100,
                           epochs = 1,
                           verbose = 1,
                           validation_data = (cv_images, cv_labels),
                           callbacks = callbacks)


# In[20]:

predictions = model.predict_classes(test_images.reshape(test_images.shape[0], 28, 28, 1), verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("predictions.csv", index=False, header=True)


# In[ ]:

