#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:03:16 2018

@author: ninguem
"""

import sys
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

tf.executing_eagerly()

BATCH_SIZE = 10
MAX_EPOCH = 60000

NFeatures = 8


os.makedirs('figs', exist_ok=True)
os.makedirs('saved', exist_ok=True)



pathGenerator = 'saved\\modelGenerator'
pathDiscriminator = 'saved\\modelDiscriminator'


lossFnc = tf.keras.losses.mean_squared_error

optimizerDiscirminator = tf.keras.optimizers.Adam(learning_rate = 0.001)
optimizerGenerator     = tf.keras.optimizers.Adam(learning_rate = 0.0001)
 
optimizerDiscirminator = tf.keras.optimizers.RMSprop(learning_rate = 0.0001)
optimizerGenerator     = tf.keras.optimizers.RMSprop(learning_rate = 0.00001)




folderName = './emojis/'
inputImgSize = (56, 56)
inputImgSize2 = 14

X_train = np.ndarray((0,inputImgSize[0],inputImgSize[1],3))

files = os.listdir(folderName)
for file in files:
    name, ext = file.split('.')

    if (ext == 'png'):
        
        digit, _ = name.split('-')


        _I = plt.imread(folderName+file).reshape((1,inputImgSize[0],inputImgSize[1],3))
    
        X_train = np.concatenate((X_train, _I), 0)
    


NTrain = X_train.shape[0]



initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.03)


# build the discriminator model
modelDiscriminator = tf.keras.models.Sequential()
modelDiscriminator.add( tf.keras.layers.Input(shape=(inputImgSize[0], inputImgSize[1], 3)) )
modelDiscriminator.add( tf.keras.layers.Conv2D(64, (7,7), padding="same", kernel_initializer = initializer ) )
modelDiscriminator.add( tf.keras.layers.BatchNormalization() )
modelDiscriminator.add( tf.keras.layers.LeakyReLU() )

modelDiscriminator.add( tf.keras.layers.MaxPooling2D((2,2)) )

# modelDiscriminator.add( tf.keras.layers.Conv2D(64, (3,3), padding="same", kernel_initializer = initializer ) )
# modelDiscriminator.add( tf.keras.layers.BatchNormalization() )
# modelDiscriminator.add( tf.keras.layers.LeakyReLU() )

modelDiscriminator.add( tf.keras.layers.MaxPooling2D((2,2)) )

modelDiscriminator.add( tf.keras.layers.Conv2D(32, (3,3), padding="same", kernel_initializer = initializer ) )
modelDiscriminator.add( tf.keras.layers.BatchNormalization() )
modelDiscriminator.add( tf.keras.layers.LeakyReLU() )

modelDiscriminator.add( tf.keras.layers.Flatten() )
modelDiscriminator.add( tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer = initializer) )



def computeDiscriminatorLoss(labelFromRealImage, labelFromGeneratedImage):
    generatedImageLoss = tf.losses.binary_crossentropy(tf.zeros(labelFromGeneratedImage.shape), labelFromGeneratedImage)
    realImageLoss = tf.losses.binary_crossentropy(tf.ones(labelFromRealImage.shape), labelFromRealImage)
    
    return tf.reduce_mean( generatedImageLoss + realImageLoss )



#M = tf.keras.Model(modelDiscriminator.inputs, modelDiscriminator.layers[1].output,  modelDiscriminator.layers[2].output )

# FEATURES!!!


modelGenerator = tf.keras.Sequential()
modelGenerator.add( tf.keras.layers.Dense(inputImgSize[0]*inputImgSize[1]*16, use_bias=False, input_shape=(NFeatures,) ) )
modelGenerator.add( tf.keras.layers.BatchNormalization() )
modelGenerator.add( tf.keras.layers.LeakyReLU() )

modelGenerator.add( tf.keras.layers.Reshape( (int(inputImgSize[0]/4), int(inputImgSize[1]/4), 16*16)) )

# modelGenerator.add( tf.keras.layers.Conv2DTranspose(128, (9, 9), strides=(1, 1), padding='same', use_bias=False) )
# modelGenerator.add( tf.keras.layers.BatchNormalization() )
# modelGenerator.add( tf.keras.layers.LeakyReLU() )

modelGenerator.add( tf.keras.layers.Conv2DTranspose(512, (7, 7), strides=(1, 1), padding='same', use_bias=False) )
modelGenerator.add( tf.keras.layers.BatchNormalization() )
modelGenerator.add( tf.keras.layers.LeakyReLU() )


modelGenerator.add( tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False) )
modelGenerator.add( tf.keras.layers.BatchNormalization() )
modelGenerator.add( tf.keras.layers.LeakyReLU() )

modelGenerator.add( tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid') ) 

    
#modelGenerator.add( tf.keras.layers.Conv2D(3, kernel_size=(1,1), activation='sigmoid', padding='same', kernel_initializer = initializer) ) 


def computeGeneratorLoss(labelFromGeneratedImage):
    generatedImageLoss = tf.losses.binary_crossentropy( tf.ones(labelFromGeneratedImage.shape), labelFromGeneratedImage )
    
    return tf.reduce_mean( generatedImageLoss )



# load last 
# modelDiscriminator = tf.keras.models.load_model(pathDiscriminator)
# modelGenerator = tf.keras.models.load_model(pathGenerator)





_GENLoss = []
_DISCLoss = []

saveCounter = 0

featureFigs = np.random.randn(1,NFeatures)

NSwitch = 1
trainDiscriminatorOnly = True

for epoch in range(MAX_EPOCH):
    print('epoch : %d of %d'%(epoch, MAX_EPOCH))

    p = np.random.permutation(NTrain)

    X_train = X_train[p]

    X_train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size=BATCH_SIZE, drop_remainder=True)
    
    i = 0    
   
    for X in X_train_dataset:

        with tf.GradientTape() as tapeGenerator, tf.GradientTape() as tapeDiscrinimator:
            
            tapeGenerator.watch( modelGenerator.trainable_variables )
            tapeDiscrinimator.watch( modelDiscriminator.trainable_variables )
            
            realImages = X
            generatedImages = modelGenerator( np.random.randn(BATCH_SIZE,NFeatures) )

            discriminatorPredictionsForRealImages = modelDiscriminator(realImages)
            discriminatorPredictionsForGeneratedImages = modelDiscriminator(generatedImages)
            
            discriminatorLoss = computeDiscriminatorLoss(discriminatorPredictionsForRealImages, discriminatorPredictionsForGeneratedImages)
            generatorLoss = computeGeneratorLoss(discriminatorPredictionsForGeneratedImages)


            gradsGenerator = tapeGenerator.gradient(generatorLoss, modelGenerator.trainable_variables)
            gradsDiscriminator = tapeDiscrinimator.gradient(discriminatorLoss, modelDiscriminator.trainable_variables)
    
            # if not trainDiscriminatorOnly:
            #     optimizerDiscirminator.apply_gradients(zip(gradsDiscriminator, modelDiscriminator.trainable_variables))
            #     print('train discriminator')
            # else:
            #     optimizerGenerator.apply_gradients(zip(gradsGenerator, modelGenerator.trainable_variables))
            #     print('train generator')

            optimizerDiscirminator.apply_gradients(zip(gradsDiscriminator, modelDiscriminator.trainable_variables))
            optimizerGenerator.apply_gradients(zip(gradsGenerator, modelGenerator.trainable_variables))
                
        print('batch : %d of %d (discriminatorLoss = %.4f, generatorLoss = %.4f)'%(i,int(NTrain/BATCH_SIZE),discriminatorLoss, generatorLoss ))
    
        i = i + 1


    if epoch % NSwitch == 0:
        trainDiscriminatorOnly = not trainDiscriminatorOnly

        
        
    _GENLoss.append(generatorLoss.numpy())
    _DISCLoss.append(discriminatorLoss.numpy())

    if (epoch%10 == 0):
        plt.subplot(1,2,1)
        plt.cla()
        plt.plot(_GENLoss)
        plt.plot(_DISCLoss)
        plt.pause(0.0001)

    if (epoch%100 == 0):
        plt.subplot(1,2,2)
        plt.cla();
        plt.imshow( modelGenerator.predict(featureFigs).reshape((inputImgSize[0],inputImgSize[1],3)) )

        plt.savefig('figs\\fig%d.png'%saveCounter)
        
        saveCounter += 1


plt.figure()
c = 1
for i in range(5):
    for j in range(5):
        plt.subplot(5,5,c)
        plt.imshow( modelGenerator.predict(np.random.randn(1,NFeatures)).reshape(inputImgSize[0],inputImgSize[1],3)  )
        c = c+1


modelGenerator.save(pathGenerator)
modelDiscriminator.save(pathDiscriminator)






