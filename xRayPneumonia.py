import os
import json
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 24
AUTOTUNE = tf.data.AUTOTUNE

# download kaggle api & data 
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp 'd/MyDrive/kaggle/kaggle.json' ~/.kaggle/
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip
!ls

# form tf datasets
trainDs = tf.keras.preprocessing.image_dataset_from_directory(
  'chest_xray/chest_xray/train',
  validation_split=.2,
  subset='training',
  seed=123,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE)

valDs = tf.keras.preprocessing.image_dataset_from_directory(
  'chest_xray/chest_xray/train',
  validation_split=.2,
  subset='validation',
  seed=123,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE)

classNames = trainDs.class_names
print(classNames)

# check & visualize ds
plt.figure(figsize=(10, 10))
for images, labels in trainDs.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(classNames[labels[i]])

for imgBatch, labelBatch in trainDs:
    print(imgBatch.shape, labelBatch.shape)
    break

# prepare dataset for the fit 
trainDs = trainDs.cache().prefetch(buffer_size=AUTOTUNE)
valDs = valDs.cache().prefetch(buffer_size=AUTOTUNE)

# fit (scaling as the layer)
model1 = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(classNames))
])

model1.compile(
    optimizer='adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model1.fit(
    trainDs,
    validation_data = valDs,
    epochs = 7
)

model2 = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(classNames))
])

model2.compile(
    optimizer='adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model2.fit(
    trainDs,
    validation_data = valDs,
    epochs = 10
)

# save  the model
MODEL_DIR = '/content/tmp'

version = 1
exportPath = os.path.join(MODEL_DIR, str(version))
print(exportPath)

if os.path.isdir(exportPath):
    !rm -r {exportPath}

tf.keras.models.save_model(
    model1, 
    exportPath,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

version = 2
exportPath = os.path.join(MODEL_DIR, str(version))
print(exportPath)

if os.path.isdir(exportPath):
    !rm -r {exportPath}

tf.keras.models.save_model(
    model2, 
    exportPath,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

# outload the model
!zip -r tmp.zip tmp
from google.colab import files
files.download('tmp.zip')

# save json to predict
testFnN = glob.glob('chest_xray/chest_xray/val/NORMAL/*.jpeg')
testFnP = glob.glob('chest_xray/chest_xray/val/PNEUMONIA/*.jpeg')
testFn = testFnP[-5:] + testFnN[-5:]
tst = []
for f in testFn:
    testArray = tf.keras.preprocessing.image.load_img(f, target_size=IMAGE_SIZE)
    testArray = tf.keras.preprocessing.image.img_to_array(testArray)
    testArray = tf.expand_dims(testArray, 0)
    tst.append(testArray)
    pred = model2.predict(testArray)
    score = tf.nn.softmax(pred[0])
    print(classNames[np.argmax(score)], 100*np.max(score))

data = json.dumps(
    {'signature_name': 'serving_default',
     'instances': np.vstack(tst).tolist()} # instances to predict
)

# outload response json
jsonFile = open('predict.json', 'w')
jsonFile.write(data)
jsonFile.close()

files.download('predict.json')