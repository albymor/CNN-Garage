#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


def predict(model_path ,image_path):
  interpreter = tf.lite.Interpreter(
      model_path=model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(image_path).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data))

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])

  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]

  labels = ['close', 'open']
  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

  print('Predicted label is: {}'.format(labels[top_k[0]]))

  print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='./test.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='./model.tflite',
      help='.tflite model to be executed')

  args = parser.parse_args()

  predict(args.model_file, args.image)
