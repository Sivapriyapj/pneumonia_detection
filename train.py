#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions


# In[4]:


path = './pneumonia_detection/data/data_xray/train/NORMAL'
name = 'IM-0115-0001.jpeg'
fullname = f'{path}/{name}'
fullname


# In[5]:


img = load_img(fullname,target_size=(224,224))
img


# In[6]:


x = np.array(img)
x.shape


# ## Pre-trained conv nets

# In[7]:


model = MobileNetV2(weights='imagenet',input_shape=(224,224,3))


# In[8]:


X = np.array([x]) # since this model expects multiple images, we create an array
X.shape # gives shape as (1,299,299,3) 1 since there is only 1 image


# In[9]:


X = preprocess_input(X) #scales input btw -1 and 1


# In[10]:


pred = model.predict(X)
pred.shape


# In[11]:


decode_predictions(pred)


# ## TRANSFER LEARNING

# In[12]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[13]:


train_gen =ImageDataGenerator(preprocessing_function = preprocess_input)


# In[14]:


train_ds = train_gen.flow_from_directory('./pneumonia_detection/data/data_xray/train',
                              target_size=(150,150),
                             batch_size = 32
                             )
#making images smaller , 150 x 150 so that training is faster.


# In[15]:


#shape of the batch is : (32,150,150,150,3) so 32 vectors will be produced.


# In[16]:


get_ipython().system('ls -l pneumonia_detection/data/data_xray/train')


# In[17]:


train_ds.class_indices


# In[18]:


X,y = next(train_ds) # returns next batch of x and y i.e, features of images and y labels.


# In[19]:


X


# In[20]:


X.shape


# In[21]:


y

# 10 classes are columns are hot encoded from 0 to 9, i.e --> 1 at column 9 means tshirt etc.


# In[22]:


val_gen =ImageDataGenerator(preprocessing_function = preprocess_input)
val_ds = val_gen.flow_from_directory('./pneumonia_detection/data/data_xray/val',
                              target_size=(150,150),
                             batch_size = 32,shuffle = False
                             )
# we dont need to shuffle since it is for validating.


# ## Training the fi model

# In[61]:


def make_model(input_size=150, learning_rate=0.001, size_inner=500,
               droprate=0.8):

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    outputs = keras.layers.Dense(2)(vectors)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[62]:


input_size = 224


# In[63]:


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_ds = train_gen.flow_from_directory(
    './pneumonia_detection/data/data_xray/train',
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    './pneumonia_detection/data/data_xray/val',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


# In[64]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'MobileNetV2_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[65]:


learning_rate = 0.001
size = 500
droprate = 0.8

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=8, validation_data=val_ds,
                   callbacks=[checkpoint])


# ## Using the model

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import preprocess_input


# In[ ]:


test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    './pneumonia_detection/data/data_xray/test',
    target_size=(224, 224),
    batch_size=32,
    shuffle=False
)


# In[ ]:


model = keras.models.load_model('MobileNetV2_v4_1_07_0.938.h5')


# In[ ]:


model.evaluate(test_ds)


# In[ ]:


path = 'pneumonia_detection/data/data_xray/test/PNEUMONIA/person100_bacteria_475.jpeg.jpg'


# In[ ]:


img = load_img(path, target_size=(224, 224))


# In[ ]:


import numpy as np


# In[ ]:


x = np.array(img)
X = np.array([x])
X.shape


# In[ ]:


X = preprocess_input(X)


# In[ ]:


pred = model.predict(X)


# In[ ]:


classes = [
    'Normal',
    'Pneumonia'
]


# In[ ]:


dict(zip(classes, pred[0]))


# In[ ]:




