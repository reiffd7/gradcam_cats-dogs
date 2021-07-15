from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def build_model_top(input_shape):

  ## Extra layers we need to use to supplement the base model for classification

  ## Pre-mobilenet layers -> data augmentation and standaridation
  data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
  mnv2_preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
  rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1) 

  inp = tf.keras.layers.Input(shape=input_shape)
  x = rescale(inp)
  # x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
  # print(tf.keras.backend.is_keras_tensor(x))
  return x, inp
  
def build_model_bottom(inp, base_model, dropout_rate, base_model_trainable):
  base_model.trainable = base_model_trainable
  last_layer = base_model.get_layer(index=len(base_model.layers)-1).name
  out_relu = base_model.get_layer(last_layer)

  ## Post-mobilenet layers -> globabl average pooling & prediction
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

  x = global_average_layer(out_relu.output)
  x = tf.keras.layers.Dropout(dropout_rate)(x)
  out = prediction_layer(x)

  model = tf.keras.models.Model(inputs = inp, outputs = out)
  # base_learning_rate = learning_rate
  # model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
  #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  #               metrics=['accuracy'])



  return model



def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)


def polynomial_decay_schedule(initial_lr=1e-3, power=1.0, epochs=100):

  def schedule(epoch):
    decay = (1 - (epoch/float(epochs))) ** power
    alpha = initial_lr * decay
    return float(alpha)

  return LearningRateScheduler(schedule)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    