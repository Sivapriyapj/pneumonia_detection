#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor



interpreter = tflite.Interpreter(model_path='pneumoniadetector-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size =(224,224)) # as mobilenetv2 has similar preprocessing function to xception, but for 224 x224

#url = 'https://github.com/Sivapriyapj/pneumonia_detection/blob/main/data/data_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'

classes = [
        'NORMAL',
        'PNEUMONIA'  
    ]
def predict(url):
    X = preprocessor.from_url(url)



    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred= interpreter.get_tensor(output_index)

    

    return dict(zip(classes,pred[0]))






