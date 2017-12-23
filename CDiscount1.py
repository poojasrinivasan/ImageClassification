import numpy as np
from PIL import Image
import bson
import pandas as pd
import  random
from skimage.data import imread
import  io
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from sklearn import metrics

class MetricHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accMetric = []
        self.lossMetric = []
        
    def on_epoch_end(self, batch, logs={}):
        self.accMetric.append(logs.get('acc'))
        self.lossMetric.append(logs.get('loss'))

logging.basicConfig(filename='cnn.log', level=logging.INFO)

categories = pd.read_csv('D:\\cdiscountData\\category_names.csv',  encoding='latin1')
categories = categories.iloc[1:].values
catList = categories[:, 0].tolist()
randomCat = random.sample(catList, 5)
randomCat = list(map(int,randomCat))

data = bson.decode_file_iter(open('D:\\cdiscountData\\train.bson', 'rb'))

X = []
Y = []

for c, d in tqdm(enumerate(data), total=7069896):
   # product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    if category_id in randomCat:
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            X.append(picture)
            Y.append(randomCat.index(category_id))
            
X = np.asarray(X)
print(X.shape)

Y = np.asarray(Y)
print(Y.shape)
Y = np_utils.to_categorical(Y, num_classes=len(randomCat))
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

dataGenerator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, vertical_flip=True)
dataGenerator.fit(X_train)

nEpochs = 10
batchSize = 100

nKernels1 = 32
kernelSize1 = (5, 5)

nKernels2 = 32
kernelSize2 = (5, 5)

nKernels3 = 32
kernelSize3 = (5, 5)

poolSize1 = (3, 3)
poolSize2 = (3, 3)
poolSize3 = (2, 2)

denseLayerSize = 1000
dropout1 = 0.25
dropout2 = 0.5

classifier = Sequential()
classifier.add(Convolution2D(nKernels1, kernel_size=kernelSize1, activation='relu', input_shape=(180, 180, 3)))
classifier.add(MaxPooling2D(pool_size=poolSize1, strides=poolSize1))
classifier.add(Convolution2D(nKernels2, kernel_size=kernelSize2, activation='relu'))
classifier.add(MaxPooling2D(pool_size=poolSize2, strides=poolSize2))
classifier.add(Convolution2D(nKernels3, kernel_size=kernelSize3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=poolSize3, strides=poolSize3))
classifier.add(Flatten())
classifier.add(Dropout(dropout1))
classifier.add(Dense(denseLayerSize, activation='relu'))
classifier.add(Dropout(dropout2))
classifier.add(Dense(len(randomCat), activation='softmax'))

metricHistory = MetricHistory()

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit_generator(dataGenerator.flow(X_train, y_train, batch_size=batchSize), steps_per_epoch=len(X_train)/batchSize, epochs=nEpochs, callbacks=[metricHistory])

score = classifier.evaluate(X_test, y_test)
print('[loss, accuracy] : {}'.format(score))

y_pred = classifier.predict_proba(X_test)

auc = metrics.roc_auc_score(y_test, y_pred)
print('AUC score : {}'.format(auc))

plt.figure(1)
plt.plot([0, 1], [0, 1])
for i in range(0, len(randomCat)):
    fpr, tpr, thresholds = metrics.roc_curve([item[i] for item in y_test], [item[i] for item in y_pred])
    plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig('ROC{}.png'.format(''.join(map(str, randomCat))))

plt.figure(2)
plt.plot(range(1,nEpochs+1), metricHistory.accMetric)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('Accuracy{}.png'.format(''.join(map(str, randomCat))))

plt.figure(3)
plt.plot(range(1,nEpochs+1), metricHistory.lossMetric)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Loss{}.png'.format(''.join(map(str, randomCat))))

logging.info('Category IDs : {}\nnEpochs : {}, batchSize = {}, nKernels1 : {}, nKernels2 : {}, nKernels3 : {}, kernelSize1 : {}, kernelSize2 : {}, kernelSize3 : {}, poolSize1 : {}, poolSize2 : {}, poolSize3 : {}, denseLayerSize : {}, dropout1 : {}, dropout2 : {}\nLoss : {}, Accuracy : {}, AUC : {}'
             .format(str(randomCat), nEpochs, batchSize, nKernels1, nKernels2, nKernels3, kernelSize1, kernelSize2, kernelSize3, poolSize1, poolSize2, poolSize3, denseLayerSize, dropout1, dropout2, score[0], score[1], auc))
