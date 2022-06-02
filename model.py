import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import *
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalCrossentropy, MeanIoU
import tensorflow.keras.backend as K
import numpy as np
from tensorflow import keras

def unet(input_size, pretrained_weights=None, learning_rate=1e-4, classify_level=4, loss_type='cross_entropy'):
    """
    Model based on Unet (https://arxiv.org/abs/1505.04597) with slight modification with categorization/loss function
    Args:
    - input_size: shape of each input image (side length, side length, channel)
    - pretrained_weights: path to pretrained weights, if any
    - learning_rate: learning rate for Adam optimizer
    - classify_level: integer of how many level for HMS. 4 for density and 2 for binary.
    """

    initializer = 'he_normal'

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv10 = Conv2D(classify_level, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)



    def weightedLoss(originalLossFunc, weightsList):

        def lossFunc(true, pred):

            axis = -1 #if channels last
            #axis=  1 #if channels first


            #argmax returns the index of the element with the greatest value
            #done in the class axis, it returns the class index
            classSelectors = K.argmax(true, axis=axis)
                #if your loss is sparse, use only true as classSelectors

            #considering weights are ordered by class, for each class
            #true(1) if the class index is equal to the weight index
            one64 = np.ones(1, dtype=np.int64)
            classSelectors = [K.equal(one64[0]*i, classSelectors) for i in range(len(weightsList))]

            #casting boolean to float for calculations
            #each tensor in the list contains 1 where ground true class is equal to its index
            #if you sum all these, you will get a tensor full of ones.
            classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

            #for each of the selections above, multiply their respective weight
            weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

            #sums all the selections
            #result is a tensor with the respective weight for each element in predictions
            weightMultiplier = weights[0]
            for i in range(1, len(weights)):
                weightMultiplier = weightMultiplier + weights[i]


            #make sure your originalLossFunc only collapses the class axis
            #you need the other axes intact to multiply the weights tensor
            loss = originalLossFunc(true,pred)
            loss = loss * weightMultiplier

            return loss
        return lossFunc

    weights_hms = [0.1, 1e6, 1e6, 1e6]  #1e6
    loss_func = 'jaccard'
    if loss_type == 'cross_entropy':
        loss_func = 'sparse_categorical_crossentropy'
    elif loss_type == 'jaccard':
        loss_func = lambda y_true, y_pred: jaccard_distance(y_true, y_pred, classify_level=classify_level, smooth=100)


    model.compile(
        optimizer = Adam(lr = learning_rate),
        loss = weightedLoss(loss_func, weights_hms),
        #loss = weightedLoss(keras.losses.sparse_categorical_crossentropy, weights_hms), #cross entropy loss
        #loss = weightedLoss(jx, weights_hms), #jaccard loss
	#loss = loss_func, 
        metrics = [
            SparseTopKCategoricalAccuracy(k=1),
            SparseCategoricalCrossentropy(axis=-1),
            #MeanIoU(classify_level)
            ]
        )

    if pretrained_weights:
    	model.load_weights(pretrained_weights)

    return model


def jaccard_distance(y_true, y_pred, classify_level=4, smooth=1e-6):
    #flatten label and prediction tensors
    y_true = tf.one_hot(indices=tf.cast(y_true, tf.uint8), depth=classify_level)
    inputs = K.batch_flatten(y_pred)
    targets = K.batch_flatten(y_true)

    intersection = K.sum(targets * inputs, axis=-1, keepdims=True)
    total = K.sum(targets, axis=-1, keepdims=True) + K.sum(inputs, axis=-1, keepdims=True)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

    # y_true = tf.one_hot(indices=y_true, depth=classify_level)
    # intersection = keras.sum(y_true * y_pred, axis=-1)
    # sum_ = keras.sum(y_true + y_pred, axis=-1)
    # jac = (intersection + smooth) / (sum_ - intersection + smooth)
    # return (1 - jac) * smooth
