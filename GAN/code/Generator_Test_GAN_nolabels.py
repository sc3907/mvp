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
#from keras.utils.visualize_util import plot

import tensorflow as tf

def generator_model(input_shape, output_shape, weights_path=None):
    nb_filters = 32
    pool_size = (2, 1)
    kernel_size = (3, 1)
    #
    input_layer = Input(shape=input_shape)
    #x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same", activation="relu",dim_ordering="tf")(input_layer)
    #for i in range(2):
    #    y = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same", activation="relu")(x)
    #    y = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same")(y)
    #    x = merge([x, y], mode="sum")
    #    x = Activation("relu")(x)
    #    x = MaxPooling2D(pool_size=pool_size)(x)
    #
    f = Flatten()(input_layer)
    d1 = Dense(100,input_shape=input_shape,activation='relu')(f)
    drop1 = Dropout(0.05)(d1)
    d2 = Dense(100,activation='relu')(drop1)
    drop2 = Dropout(0.05)(d2)
    prod = np.prod(output_shape)
    final_dense = Dense(prod)(drop2)
    out_layer = Reshape(output_shape)(final_dense) 
    model = Model(input=input_layer, output=out_layer)     
    if weights_path:
        model.load_weights(weights_path)
    return model

def discriminator_model(input_shape, weights_path=None):
    nb_filters = 32
    pool_size = (2, 1)
    kernel_size = (3, 1)
    
    input_layer = Input(shape=input_shape)
    c1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', dim_ordering='tf')(input_layer)
    a1 = Activation('relu')(c1)
    c2 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            dim_ordering='tf')(a1)
    a2 = Activation('relu')(c2)
    pool = MaxPooling2D(pool_size=pool_size,
                           dim_ordering='tf')(a2)
    dropout = Dropout(0.25)(pool)
    f = Flatten()(dropout)
    #split pathogenic into 2 classes
    dense_layer = Dense(128,activation='relu')(f)
    dense_layer = Dropout(0.25)(dense_layer)
    damaging_clust = Dense(2)(dense_layer)
    damaging_clust = Dense(1)(damaging_clust)    
    benign_clust = Dense(1)(dense_layer)
    fake_clust = Dense(1)(dense_layer)
    merge_clusts = merge([benign_clust,damaging_clust,fake_clust],mode='concat')
    out_layer = Activation('softmax')(merge_clusts)
    model = Model(input=input_layer, output=out_layer)        
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    #model.compile(loss=loss_unlabeled,
    #              optimizer='adam',
    #              metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path)
    return model

def loss_unlabeled(y_true,y_pred):
    loss = 0
    loss += K.categorical_crossentropy(y_true, y_pred)       
    return loss

def neg_loss_unlabeled(y_true,y_pred):
    return -1.0*loss_unlabeled(y_true,y_pred)

def load_data(fname='../data/HS_input_data.csv'):
    '''read in data
    '''
    exclude_cols = {'target', 'CADD_phred', 'Eigen-phred', 'Eigen-PC-phred','Eigen-PC-raw_rankscore',
                    'MetaSVM_rankscore', 'MetaLR_rankscore','M-CAP_rankscore', 'DANN_rankscore','CADD_raw_rankscore',
                    'Polyphen2_HVAR_rankscore', 'MutationTaster_converted_rankscore', 
                    'FATHMM_converted_rankscore', 'fathmm-MKL_coding_rankscore',
                    '#chr','pos(1-based)', 'ref','alt','category','source','INFO','disease','genename',
                    'hg19_chr','hg19_pos(1-based)',
                    'prec','s_hat_log'              
}
    
    df = pd.read_csv(fname)
    y = np.array(df.pop('target'))
    cols = [col for col in df.columns if col not in exclude_cols]
    print 'cols used:', cols
    print '{} cols used'.format(len(cols))
    X = df[cols]
    return X.values, y

def training(argv):
    print("num args: {}".format(len(argv)))
    print("argv[0]: {}".format(argv[0]))
    print("argv[1]: {}".format(argv[1]))
    min_val_loss = sys.float_info.max
    nb_epoch = 30
    batch_size = 256
    best_weights_filepath = "GAN_acceptsunlabeled_best_cross_val.hdf5"

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    print('Loading training data...')
    #X, Y = load_data("../data/HS_input_data_unlab_1.csv") 
    X, Y = load_data("../data/HS_input_data.csv") 
    Y = np.array([-1 for i in range(X.shape[0])]) #make all unlabeled
    input_shape = (X.shape[1], 1, 1)
    # convert into 4D tensor For 2D data (e.g. image), "tf" assumes (rows, cols, channels) 
    X = X.reshape(X.shape[0], X.shape[1], 1, 1)
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)

    nb_example = X.shape[0]
    nb_features = X.shape[1]
    X_labeled = []
    X_unlabeled = []
    Y_categ_labeled = []
    Y_categ_unlabeled = []
    nb_labeled,nb_unlabeled = 0,0
    for i in range(nb_example):
        if Y[i]>0.5:
            X_labeled.append(X[i])
            Y_categ_labeled.append([0.0,1.0,0.0])
            nb_labeled += 1
        elif Y[i]<=0.5 and Y[i]>=0:
            X_labeled.append(X[i])
            Y_categ_labeled.append([1.0,0.0,0.0])
            nb_labeled += 1
        else:
            X_unlabeled.append(X[i])
            Y_categ_unlabeled.append([0.499,0.499,0.002])
            nb_unlabeled += 1
    X_labeled = np.array(X_labeled).reshape(nb_labeled,X.shape[1], 1, 1)
    X_unlabeled = np.array(X_unlabeled).reshape(nb_unlabeled,X.shape[1], 1, 1)
    Y_categ_labeled=np.array(Y_categ_labeled).reshape(nb_labeled,3)
    Y_categ_unlabeled=np.array(Y_categ_unlabeled).reshape(nb_unlabeled,3)

    split = 1
    best_acc = 0
    best_gen_score = 0
    first = True
    print X_labeled.shape
    print Y_categ_labeled.shape
    for train, test in kfold.split(X,Y):
        #print("num of train ex: %d" % nb_train_example)
        print("split: %d"%split)    
        g_input_length=50
        G = generator_model((g_input_length,1,1),input_shape,weights_path="Generator_weights_1.hdf5")
        D = discriminator_model(input_shape)
#weights_path="discriminator_weights/GAN_acceptsunlabeled_best_cross_val_first.hdf5"
        D_two_class = Sequential()
        D_two_class.add(D)
        def output_real(x):
            #return 2*K.sum(x[:,0:2],axis=1,keepdims=True)-1
            return K.sum(x[:,0:2],axis=1,keepdims=True)
        D_two_class.add(Lambda(output_real))
        def wasserstein_loss(y_true,y_pred):
            return K.mean(y_true*y_pred)
        D_two_class.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        GD = Sequential()
        GD.add(G)
        D.trainable = False
        GD.add(D_two_class)
        def neg_binary_cross_entropy(y_true,y_pred):
            return -K.binary_crossentropy(y_true, y_pred)
        GD.compile(loss=neg_binary_cross_entropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        #use with regular D_two_class

        X_train = X[train]
        for epoch in range(nb_epoch):
            a1,a2,a3,l3 = ([],[],[],[])
            shuffled = np.random.permutation([i for i in range(X_train.shape[0])])
            nb_batches = int(np.ceil(X_train.shape[0]/batch_size))
            for batch in range(nb_batches):
                D.trainable = True
                for layer in D.layers:
                    layer.trainable=True
                loss1,acc1 = D_two_class.train_on_batch(X_train[shuffled[batch*batch_size:(batch+1)*batch_size]],[[1]]*batch_size)
                noise = np.random.normal(0,1,(batch_size,g_input_length,1,1))
                generated = G.predict(noise)
                loss2,acc2 = D.train_on_batch(generated,[[0,0,1]]*batch_size)
                D.trainable = False
                for layer in D.layers:
                    layer.trainable = False
                #for i in range(3):
                noise = np.random.normal(0,1,(batch_size,g_input_length,1,1))
                loss3,acc3 = GD.train_on_batch(noise,[[0]]*batch_size)               
                a1.append(acc1)
                a2.append(acc2)
                a3.append(acc3)
                l3.append(loss3)
            print("Epoch: %d SamplesAcc: %f NoiseAcc: %f GeneratorAccLoss: %f %f" % (epoch,np.mean(a1),np.mean(a2),np.mean(a3),np.mean(l3)))
        # evaluate the model
        scores = GD.evaluate(np.random.normal(0,1,(1024,g_input_length,1,1)), [[1]]*1024, verbose=1)
        if scores[1] > best_gen_acc:
            G.save_weights("best_generator_weights_inputsize_"+str(g_input_length)+".hdf5", overwrite=True)        
        G.save_weights("Generator_weights_"+str(split)+".hdf5", overwrite=True)
        D.save_weights("Discriminator_weights_"+str(split)+".hdf5", overwrite=True)
        split += 1

def view_gen_out(weights_file,test_file,outfile):
    nb_feats=38
    X = pd.read_csv(test_file,index_col=0)
    X = X.values
    input_shape = (X.shape[1], 1, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1, 1)
    G = generator_model(input_shape,(nb_feats,1,1),weights_path=weights_file)
    predictions = G.predict(X)
    predictions = predictions.reshape(X.shape[0],nb_feats)
    X = X.reshape(X.shape[0],X.shape[1])
    out = np.hstack((X,predictions))
    print X.shape
    print predictions.shape
    print out.shape
    out_table = pd.DataFrame(out)
    out_table.to_csv(outfile,sep=",")

if __name__=="__main__":
    if sys.argv[1] == "training":
        training(sys.argv[0:])
    elif sys.argv[1] == "view_gen_out":
        view_gen_out(sys.argv[2],sys.argv[3],sys.argv[4])
