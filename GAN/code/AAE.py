#John Long

import numpy as np
seed = 7
np.random.seed(seed)
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sys

from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.core import Lambda

import tensorflow as tf
from GPU_GAN import load_data

def create_models(input_shape, num_classes, style_shape, weights_path=None):
    input_layer = Input(shape=input_shape,name='input_layer')
    """c1 = Convolution2D(32, 3, 1, border_mode="same", activation="relu",dim_ordering="tf")(input_layer)
    c2 = Convolution2D(32, 3, 1, border_mode="same", activation="relu",dim_ordering="tf")(c1)
    pool = MaxPooling2D(pool_size=(2,1),dim_ordering='tf')(c2)
    dropout = Dropout(0.1)(pool)
    f = Flatten()(pool)"""
    f = Flatten()(input_layer)
    encoder_d1 = Dense(100,activation='relu')(f)
    encoder_drop1 = Dropout(0.01)(encoder_d1)
    encoder_d2 = Dense(50,activation='relu')(encoder_drop1)
    encoder_drop2 = Dropout(0.02)(encoder_d2)
    encoder_d3 = Dense(40,activation='relu')(encoder_drop2)
    encoder_drop3 = Dropout(0.01)(encoder_d3)
    label = Dense(num_classes)(encoder_drop3)
    label_out = Activation('softmax')(label)
    style = Dense(np.prod(style_shape))(encoder_drop2)
    style_out = Reshape(style_shape)(style)
    merged_encoded = merge([label_out,style_out],mode='concat')
    encoder = Model(input=input_layer,output=[label_out,style_out])
    encoder.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
    encoder_label = Model(input=input_layer,output=label_out)
    encoder_style = Model(input=input_layer,output=style_out)
    encoder_label.summary()
    encoder_mergedout = Model(input=input_layer,output=merged_encoded)

    classifier = Model(input=input_layer, output=label_out)
    classifier.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#    decoder_d1 = Dense(40,activation='relu')(merged_encoded)
    decode_input = Input(shape=(num_classes+style_shape[0],))
    decoder_d1 = Dense(40,activation='relu')(decode_input)
    decoder_drop1 = Dropout(0.01)(decoder_d1)
    decoder_d2 = Dense(50,activation='relu')(decoder_drop1)
    decoder_drop2 = Dropout(0.02)(decoder_d2)
    decoder_d3 = Dense(100,activation='relu')(decoder_drop2)
    decoder_drop3 = Dropout(0.01)(decoder_d3)
    decoder_d4 = Dense(np.prod(input_shape))(decoder_drop3)
    auto_out = Reshape(input_shape)(decoder_d4)
    decoder = Model(input=decode_input, output=auto_out)
#    autoencoder = Model(input=input_layer,output=auto_out)
    autoencoder = Sequential()
    autoencoder.add(encoder_mergedout)
    autoencoder.add(decoder)
    autoencoder.compile(loss='mse',
                        optimizer='adam',
                        metrics=['mae'])
    
    il_label = Input(shape=(num_classes,))
    label_disc_d1 = Dense(30,activation='relu')(il_label)
    label_disc_drop1 = Dropout(0.01)(label_disc_d1)
    label_disc_out = Dense(1,name='label_disc_out')(label_disc_drop1)
        
    il_style = Input(shape=style_shape)
    style_disc_d1 = Dense(50,activation='relu')(il_style)
    style_disc_drop1 = Dropout(0.02)(style_disc_d1)
    style_disc_d2 = Dense(50,activation='relu')(style_disc_drop1)
    style_disc_drop2 = Dropout(0.02)(style_disc_d2)
    style_disc_out = Dense(1,name='style_disc_out')(style_disc_drop2)

    label_disc = Model(input=il_label, output=label_disc_out)
    label_disc.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    label_disc.summary()
    style_disc = Model(input=il_style, output=style_disc_out)
    style_disc.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    
    label_discriminator_noise = Sequential()
    label_discriminator_noise.add(encoder_label)
    label_discriminator_noise.add(label_disc)
    label_discriminator_noise.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    style_discriminator_noise = Sequential()
    style_discriminator_noise.add(encoder_style)
    style_discriminator_noise.add(style_disc)
    style_discriminator_noise.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    def gen_loss(y_true,y_pred):
        return -K.categorical_crossentropy(y_true, y_pred)       
    
    for layer in label_disc.layers:
        layer.trainable=False
    label_generator = Sequential()
    label_generator.add(encoder_label)
    label_generator.add(label_disc)
    label_generator.compile(loss=gen_loss,
                        optimizer='adam',
                        metrics=['accuracy'])
    for layer in style_disc.layers:
        layer.trainable=False
    style_generator = Sequential()
    style_generator.add(encoder_style)
    style_generator.add(style_disc)
    style_generator.compile(loss=gen_loss,
                        optimizer='adam',
                        metrics=['accuracy'])
    return encoder,decoder,classifier,autoencoder,label_disc,style_disc,label_discriminator_noise,style_discriminator_noise,label_generator,style_generator
    
def training(argv):
    min_val_loss = sys.float_info.max
    nb_epoch = 200
    batch_size = 30000
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    latent_shape = (30,)

    print('Loading training data...')
    X, Y, X_labeled, Y_categ_labeled, X_unlabeled, Y_categ_unlabeled = load_data("../data/HS_input_data.csv") 
    Y_lab_dellast = [Y_categ_labeled[i][0:2] for i in range(Y_categ_labeled.shape[0])]
    Y_categ_labeled = np.array(Y_lab_dellast)    
    input_shape = (X.shape[1], 1, 1)
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    print('X_labeled shape:', X_labeled.shape)
    print('Y_categ_labeled.shape:', Y_categ_labeled.shape)

    split = 1
    best_acc_d = 0
    best_acc_g = 0
    first = True
    cvscores = []
    for train, test in kfold.split(X_labeled,Y_categ_labeled[:,0]):
        (encoder,decoder,classifier,autoencoder,label_disc,style_disc,
        label_discriminator_noise,style_discriminator_noise,label_generator,
        style_generator) = create_models(input_shape, 2, latent_shape)
        
        X_train = X_labeled[train]
        X_train_labeled = X_labeled[train]
        X_train = np.vstack((X_train,X_unlabeled))
        print(X_train.shape)
        Y_train = Y_categ_labeled[train]
        Y_train_labeled = Y_categ_labeled[train]
        #Y_train = np.vstack((Y_train,Y_categ_unlabeled))
        relabeled_Y = Y_train.copy()
        print relabeled_Y.shape
        misclass_counts = np.array([0 for i in range(len(Y_train))])
        unlabeled_indices = []
        labeled_indices = []
        for i in range(X_train.shape[0]):
            if np.allclose(relabeled_Y[i], [0.499,0.499]):
                unlabeled_indices.append(i)  
            else:
                labeled_indices.append(i)   
        labeled_indices_atstart = list(labeled_indices)
        
        for epoch in range(nb_epoch):
            l1,a2,a3,a4 = ([],[],[],[])
            shuffled = np.random.permutation([i for i in range(X_train_labeled.shape[0])])
            nb_batches = int(np.ceil(X_train_labeled.shape[0]/batch_size))
            shuffled_bigset = np.random.permutation([i for i in range(X_train.shape[0])])
            batch_size_bigset = X_train.shape[0]/nb_batches
            for batch in range(nb_batches):
                loss1,acc1 = autoencoder.train_on_batch(X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]],X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]])
                #train discriminators to recognize true vs fake
                for layer in encoder.layers:
                    layer.trainable=False
                for layer in label_disc.layers:
                    layer.trainable=True
                for layer in style_disc.layers:
                    layer.trainable=True
                #real dist p which is label dist and normal
                loss21,acc21 = label_disc.train_on_batch(Y_train_labeled[shuffled[batch*batch_size:(batch+1)*batch_size]], [[1]]*batch_size)
                loss22,acc22 = style_disc.train_on_batch(np.random.normal(0,1,(batch_size,latent_shape[0])), [[1]]*batch_size)
                #latent dist tries to be like p
                label_discriminator_noise.train_on_batch(X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]],[[0]]*batch_size_bigset)
                style_discriminator_noise.train_on_batch(X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]],[[0]]*batch_size_bigset)
                #train generator
                for layer in encoder.layers:
                    layer.trainable=True
                for layer in label_disc.layers:
                    layer.trainable=False
                for layer in style_disc.layers:
                    layer.trainable=False
                loss23,acc23 = label_generator.train_on_batch(X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]],[[0]]*batch_size_bigset) 
                loss24,acc24 = style_generator.train_on_batch(X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]],[[0]]*batch_size_bigset)           
                loss3,acc3 = classifier.train_on_batch(X_train_labeled[shuffled[batch*batch_size:(batch+1)*batch_size]], Y_train_labeled[shuffled[batch*batch_size:(batch+1)*batch_size]])
                l1.append(loss1)
                a2.append(np.mean([acc21,acc22]))
                a3.append(np.mean([acc23,acc24]))
                a4.append(acc3)
            #classifier.evaluate(X_train_labeled,Y_train_labeled)
            #autoencoder.evaluate(X_train_labeled,X_train_labeled)
            print("Epoch: %d AutoLoss: %f NoiseAcc: %f GeneratorAccLoss: %f Classification: %f" % (epoch,np.mean(l1),np.mean(a2),np.mean(a3),np.mean(a4)))
        # evaluate the model
        scores = classifier.evaluate(X_labeled[test], Y_categ_labeled[test], verbose=1)
        print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        if scores[1] > best_acc_d:
            classifier.save_weights("best_classifier_weights_split"+str(split)+"_AAE.hdf5", overwrite=True)
        scores = autoencoder.evaluate(X_labeled[test], X_labeled[test], verbose=1)
        print("autoencoder %s: %.2f%%" % (autoencoder.metrics_names[0],scores[0]))
        #if scores[1] > best_acc_g:
        #    dec.save_weights("best_generator_weights_inputsize_"+str(latent_shape[0])+"_split"+str(split)+"_AAE.hdf5", overwrite=True)        
        decoder.save_weights("Generator_weights_"+str(split)+"_AAE.hdf5", overwrite=True)
        classifier.save_weights("Classifier_weights_"+str(split)+"_AAE.hdf5", overwrite=True)
        split += 1

def predict_to_file(weights_file,test_file,outfile):
    #print(weights_file)
    #print(test_file)
    X, Y = load_data(fname=test_file) 
    input_shape = (X.shape[1], 1, 1)
    # convert into 4D tensor For 2D data (e.g. image), "tf" assumes (rows, cols, channels) 
    X = X.reshape(X.shape[0], X.shape[1], 1, 1)
    #print('X shape:', X.shape)

    Y_categ = []
    for i in range(X.shape[0]):
        if Y[i]>0.5:
            Y_categ.append([0.0,1.0])
        else:
            Y_categ.append([1.0,0.0])
    Y_categ=np.array(Y_categ)

    enc = encoder(input_shape,(30,),weights_file)
    classifier = encoder_classifier(input_shape,enc)
    predictions = D.predict(X)
    predictions = predictions[:,1:2] / np.sum(predictions[:,0:2],axis=1).reshape(-1,1)
    predictions = predictions.T[0]
    input_data = pd.read_csv(test_file)
    input_data['GAN_prob'] = pd.Series(predictions)
    input_data.to_csv(outfile,sep=",")

if __name__=="__main__":
    if sys.argv[1] == "training":
        training(sys.argv[0:])
    elif sys.argv[1] == "test_on_file":
        test_on_file(sys.argv[2],sys.argv[3],verbose=True)
    elif sys.argv[1] == "predict_to_file":
        predict_to_file(sys.argv[2],sys.argv[3],sys.argv[4])
    elif sys.argv[1] == "view_gen_out":
        view_gen_out(sys.argv[2],sys.argv[3],sys.argv[4])
