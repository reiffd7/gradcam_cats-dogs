from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from tqdm import tqdm
from random import shuffle 
import cv2
import os



def process_data(data_dir, dog_image_list, cat_image_lst, IMG_SIZE):
  ## Helper for manual_pre_process
  data_df = []
  labels = []
  cat_count, dog_count = 0, 0
  DATA_FOLDER = data_dir
  for img in tqdm(dog_image_list):
    path = os.path.join(DATA_FOLDER, img)
    label = 1
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data_df.append([np.array(img), np.array(label), path])
  for img in tqdm(cat_image_lst):
    path = os.path.join(DATA_FOLDER, img)
    label = 0
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data_df.append([np.array(img), np.array(label), path])
  # DATA_FOLDER = aug_dir
  # for img in tqdm(dog_aug_lst):
  #   path = os.path.join(DATA_FOLDER, img)
  #   label = 1
  #   img = cv2.imread(path, cv2.IMREAD_COLOR)
  #   img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  #   data_df.append([np.array(img), np.array(label), path])
  # for img in tqdm(cat_aug_lst):
  #   path = os.path.join(DATA_FOLDER, img)
  #   label = 0
  #   img = cv2.imread(path, cv2.IMREAD_COLOR)
  #   img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  #   data_df.append([np.array(img), np.array(label), path])
  shuffle(data_df)
  return data_df


def manual_pre_process(data_dir, IMG_SIZE, DATA_SAMPLE_SIZE,  isTrain=True):
  dog_image_lst = [file for file in os.listdir(data_dir) if 'dog' in file][:int(DATA_SAMPLE_SIZE/2)]
  cat_image_lst = [file for file in os.listdir(data_dir) if 'cat' in file][:int(DATA_SAMPLE_SIZE/2)]
  # dog_aug_lst = [file for file in os.listdir(aug_dir) if 'dog' in file][:int(AUG_SAMPLE_SIZE/2)]
  # cat_aug_lst = [file for file in os.listdir(aug_dir) if 'cat' in file][:int(AUG_SAMPLE_SIZE/2)]
  # dog_image_lst = [file for file in os.listdir(dir) if 'dog' in file]
  # cat_image_lst = [file for file in os.listdir(dir) if 'cat' in file]
  data_df = process_data(data_dir, dog_image_lst, cat_image_lst, IMG_SIZE)
  X = np.array([i[0] for i in data_df]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  y = np.array([i[1] for i in data_df])
  files = np.array([i[2] for i in data_df])
  return X, y, files



class DatasetSequence(Sequence):
  ## Take the processed data and make it easiy digestible for model training

  def __init__(self, x_set, y_set, batch_size, augmentations=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

  def __len__(self):
      return int(np.ceil(len(self.x) / float(self.batch_size)))

  def __getitem__(self, idx):
      batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
      batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
      
      if self.augment == None:
        return batch_x, batch_y
      else:
        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)