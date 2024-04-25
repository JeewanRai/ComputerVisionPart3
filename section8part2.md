## Keras Convolution Neural Netowrk with MNIST
```Python
#loading MNIST dataset
from keras.datasets import mnist

# load MNIST data using tuple unpacking
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# To see data we use matplot library
import matplotlib.pyplot as plt

# check the size of the data.
x_train.shape

# grab the single image from the first index
single_image = x_train[0]
single_image
```
Output:                                   
```Python
# Loading MNIST dataset
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 107s 9us/step

# size of the data, no color channel is there.
(60000, 28, 28)

# single image from the first index
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
```
```Python
plt.imshow(single_image, cmap='gray_r')
```
Output:                                         
![alt text](image-213.png)

```Python
y_train

y_train.shape
```
Output:                                         
```Python
# feeding the data in this format/category data in this way the network will not understand
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

(60000,)
```

```Python
from keras.utils import to_categorical

# convert the labels into categorical data, 10 indicates number of category, need to encode the network can understand
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

y_test_categorical

# counting from first left side towards 0 at position 5 we have 1 which means number 5
y_train_categorical[0]

# processing x data, its not normalize, should normalize
single_image.max()

x_train = x_train / x_train.max() # same as dividing by /255

```
```Python
# y_test_categorical
array([[0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])

# y_train_categorical[0], works well with sigmoid function
 array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])      

 # single_image.max() so the value is within 0 and 1. The value should be between 0 and 1, dividing by max value
 255.0
```

```Python
scaled_image = x_train[0]
scaled_image

scaled_image.max()

# showing image after scaling has no difference than first image.

```
```Python
# The values falls between 0 and 1

# scaled_image.max()
1.0
```
Output:                                             
![alt text](image-214.png)

```Python
# without color channel
x_train.shape

x_train = x_train.reshape(60000, 28, 28, 1)
x_train.shape

x_test = x_test.reshape(10000, 28, 28, 1)
x_test.shape
```
Output:                                             
```Python
(60000, 28, 28)

# x_train after reshaping
(60000, 28, 28, 1)

# x_test after reshaping
(10000, 28, 28, 1)
```
```Python
from keras.models import Sequential

# flatten outs 2D images
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# create model, creating sequencial onject
model = Sequential()

# CONVOLUTIONAL LAYERS
# for this images we can use 32 as difault filter otherwise for complex image we need to use higher filter size
model.add(Conv2D(filters = 32, kernel_size = (4,4),input_shape = (28, 28, 1), activation='relu'))

# POOLING LAYERS
model.add(MaxPooling2D(pool_size=(2,2)))

# converting Convolution and MAXPooling layers such that single dense layer can understand
model.add(Flatten()) # 2D------> 1D

# DENSE LAYERS
model.add((Dense(128, activation='relu'))) # fully connected layer, experiment with 128 values

model.add(Dense(10, activation='softmax')) # output layer, 10 is difault value for 10 digits

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


model.fit(x_train, y_train_categorical, epochs=10)

# grab and evaluate the model
model.metrics_names
```
Output:   
# summary                                           
![alt text](image-215.png)

```Python
# epoch 
Epoch 1/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 37s 18ms/step - accuracy: 0.9159 - loss: 0.2770
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 26s 14ms/step - accuracy: 0.9852 - loss: 0.0501
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 20s 11ms/step - accuracy: 0.9907 - loss: 0.0313
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 21s 11ms/step - accuracy: 0.9936 - loss: 0.0197
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 21s 11ms/step - accuracy: 0.9951 - loss: 0.0152
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 21s 11ms/step - accuracy: 0.9970 - loss: 0.0099
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 21s 11ms/step - accuracy: 0.9974 - loss: 0.0080
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 22s 11ms/step - accuracy: 0.9984 - loss: 0.0054
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 22s 12ms/step - accuracy: 0.9988 - loss: 0.0043
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 22s 12ms/step - accuracy: 0.9990 - loss: 0.0033
<keras.src.callbacks.history.History at 0x25f3bf58950>

# output of grab and evaluate the model
['loss', 'compile_metrics']
```
```Python
# Evaluating on test data and its able to predict 98% accurately in test data.
model.evaluate(x_test, y_test_categorical)

#predict class using images not seen by model before
from sklearn.metrics import classification_report

predictions = model.predict(x_test)

y_test_categorical

predictions

y_test

# cannot compare two datasets of differet types so converting to one format
import numpy as np

# Convert the probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Now you can use 'predicted_classes' with 'y_test' in your classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_classes))
```
Output:                                 
```Python
# if the model is performing good with training data and bad with test data, its due to overfitting of the data.
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9863 - loss: 0.0638
[0.05277801677584648, 0.9887999892234802]


#y_test class
array([[0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])

 # predictions
 array([[9.2297792e-15, 1.7292027e-16, 7.4872232e-13, ..., 1.0000000e+00,
        2.2998750e-16, 8.1549677e-11],
       [2.2477912e-15, 4.2502259e-11, 1.0000000e+00, ..., 3.1703617e-25,
        9.9039379e-19, 1.1312233e-21],
       [5.0093959e-14, 1.0000000e+00, 3.0431819e-15, ..., 1.5723017e-13,
        1.9121931e-11, 4.0552490e-17],
       ...,
       [8.6883109e-25, 7.8078731e-14, 9.1847099e-24, ..., 8.3248919e-14,
        1.0957316e-13, 1.3460309e-13],
       [1.3640056e-21, 1.1529276e-21, 8.8102166e-28, ..., 2.4160694e-21,
        4.9079452e-10, 4.0378555e-23],
       [5.1946228e-17, 3.4413247e-20, 4.0085958e-19, ..., 2.1400377e-28,
        1.2465031e-20, 4.5787739e-22]], dtype=float32)

#y_test
array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

 precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.99      0.98      0.99      1032
           3       1.00      0.99      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.99      0.99      0.99       892
           6       0.98      0.99      0.98       958
           7       1.00      0.98      0.99      1028
           8       0.98      0.99      0.99       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

```
## Keras Convolution Neural Netowrk with CIFAR-10, COLOR IMAGES
```Python
from keras.datasets import cifar10

# loead data, using tuple upacking
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train.shape
x_train.shape

# x train data at position 0
x_train[0]

# x_train data shape
x_train.shape
```
Output:                                         
```Python
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 8635s 51us/step

# x_train.shape
(50000, 32, 32, 3)

# x_train data at position 0
array([[[ 59,  62,  63],
        [ 43,  46,  45],
        [ 50,  48,  43],
        ...,
        [158, 132, 108],
        [152, 125, 102],
        [148, 124, 103]],

       [[ 16,  20,  20],
        [  0,   0,   0],
        [ 18,   8,   0],
        ...,
        [123,  88,  55],
        [119,  83,  50],
        [122,  87,  57]],

       [[ 25,  24,  21],
        [ 16,   7,   0],
        [ 49,  27,   8],
        ...,
        [118,  84,  50],
        [120,  84,  50],
        [109,  73,  42]],

       ...,
...
        [179, 142,  87],
        ...,
        [216, 184, 140],
        [151, 118,  84],
        [123,  92,  72]]], dtype=uint8)

# x_train data shape
(50000, 32, 32, 3)
```
```Python
import matplotlib.pyplot as plt
plt.imshow(x_train[0])

# pre processing
x_train.max()

x_train = x_train / x_train.max() # same as dividing by /255
x_test = x_test / x_test.max() # same as dividing by /255

x_test.shape

# y_train is in integer format need to do One_hot encoding
y_train

from keras.utils import to_categorical
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)
```
Output:                       
![alt text](image-216.png)

```Python
# pre processing
255

# x_test
(10000, 32, 32, 3)

#y_train data
array([[6],
       [9],
       [9],
       ...,
       [9],
       [1],
       [1]], dtype=uint8)
```
```Python
# Build our model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# CONVOLUTIONAL LAYERS
model.add(Conv2D(filters = 32, kernel_size = (4,4), input_shape = (32, 32, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# CONVOLUTION LAYERS, the image is complex so another convolution layer is good idea
model.add(Conv2D(filters = 32, kernel_size = (4,4), input_shape = (32, 32, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # 2D------> 1D

# DENSE LAYERS
model.add(Dense(256, activation='relu')) # fully connected layer, experiment with 128 values

model.add(Dense(10, activation='softmax')) # output layer, 10 is difault value for 10 digits

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
```
Output:                               
```Python
**┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_2 (Conv2D)               │ (None, 29, 29, 32)     │         1,568 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 14, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 11, 11, 32)     │        16,416 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 5, 5, 32)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_2 (Flatten)             │ (None, 800)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 256)            │       205,056 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 10)             │         2,570 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 225,610 (881.29 KB)
 Trainable params: 225,610 (881.29 KB)
 Non-trainable params: 0 (0.00 B)
```
```Python
model.fit(x_train, y_train_categorical, verbose = 1, epochs=10)

# evaluate our model
model.metrics_names

# evaluate our model
model.evaluate(x_test, y_test_categorical)

import numpy as np

# Convert the probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Now you can use 'predicted_classes' with 'y_test' in your classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_classes))
```
Output:                                           
```Python
#epochs
Epoch 1/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 30s 17ms/step - accuracy: 0.3663 - loss: 1.7403
Epoch 2/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 29s 19ms/step - accuracy: 0.5754 - loss: 1.1998
Epoch 3/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 33s 21ms/step - accuracy: 0.6462 - loss: 1.0162
Epoch 4/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 33s 21ms/step - accuracy: 0.6925 - loss: 0.8861
Epoch 5/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 31s 20ms/step - accuracy: 0.7249 - loss: 0.8090
Epoch 6/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 33s 21ms/step - accuracy: 0.7497 - loss: 0.7347
Epoch 7/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 31s 20ms/step - accuracy: 0.7694 - loss: 0.6799
Epoch 8/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 29s 19ms/step - accuracy: 0.7818 - loss: 0.6256
Epoch 9/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 27s 17ms/step - accuracy: 0.8066 - loss: 0.5760
Epoch 10/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 31s 20ms/step - accuracy: 0.8209 - loss: 0.5295
<keras.src.callbacks.history.History at 0x25f6a250350>

# evaluate our model
['loss', 'compile_metrics']

# evaluated model with 62% accuracy
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.6261 - loss: 1.2425
[1.2423770427703857, 0.6243000030517578]

#evaluation
              precision    recall  f1-score   support

           0       0.62      0.74      0.68      1000
           1       0.57      0.85      0.68      1000
           2       0.70      0.36      0.48      1000
           3       0.55      0.26      0.35      1000
           4       0.57      0.63      0.60      1000
           5       0.53      0.60      0.56      1000
           6       0.66      0.76      0.71      1000
           7       0.79      0.65      0.71      1000
           8       0.59      0.84      0.69      1000
           9       0.77      0.56      0.65      1000

    accuracy                           0.62     10000
   macro avg       0.64      0.62      0.61     10000
weighted avg       0.64      0.62      0.61     10000
```

# Python for Computer Vision with OpenCV and Deep Learning

When ever building our own model we should have folder containing train and test data with different classes in each folder/should have each category of data.

```Python
import cv2
import matplotlib.pyplot as plt

cat_img = cv2.imread('P:/Pi OpenCV  programming/CATS_DOGS/CATS_DOGS/train/CAT/4.jpg')
cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)

plt.imshow(cat_img)

#shape of the image
cat_img.shape
```
Output:                                           
![alt text](image-217.png)

```Python
#shape of the image
(375, 500, 3)
```
Loading dog image
```Python
dog_img = cv2.imread('P:/Pi OpenCV  programming/CATS_DOGS/CATS_DOGS/train/DOG/2.jpg')
dog_img = cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB)

plt.imshow(dog_img)
```

Output:                                           
![alt text](image-218.png)

Size of dog image and cat image are not same. In the real word its not possible to get image of the same size, therefore, need to prepare data and keras has imagedatagenerator function. Also need to do transformation like rotate the image.
```Python
(199, 188, 3)
```