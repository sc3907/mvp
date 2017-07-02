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
    
    input_layer = Input(shape=input_shape)
    x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same", activation="relu")(input_layer)
    for i in range(2):
        y = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same", activation="relu")(x)
        y = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same")(y)
        x = merge([x, y], mode="sum")
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

    f = Flatten(input_shape=input_shape)(x)
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
    X, Y = load_data("../data/HS_input_data.csv") 
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
    first = True
    print X_labeled.shape
    print Y_categ_labeled.shape
    for train, test in kfold.split(X_labeled,Y_categ_labeled[:,0]):
        #print("num of train ex: %d" % nb_train_example)
        print("split: %d"%split)    
        #model = cnn_model(input_shape)
        g_input_length=50
        G = generator_model((g_input_length,1,1),input_shape)
        #if first:
        #    plot(G, show_shapes=True,to_file='../models/GAN_generator.png')
        D = discriminator_model(input_shape)
        #if first:
        #    plot(D, show_shapes=True,to_file='../models/GAN_discriminator.png')

        D_two_class = Sequential()
        D_two_class.add(D)
        def output_real(x):
            #y=list(range(K.shape(x)[0]))
            #for i,x_i in enumerate(x):
            #   y[i]=1-x_i[2]
            #return y

            #return [1-x_i[2] for x_i in x]
            #return 2*K.sum(x[:,0:2],axis=1,keepdims=True)-1
            return K.sum(x[:,0:2],axis=1,keepdims=True)
        D_two_class.add(Lambda(output_real))
        def wasserstein_loss(y_true,y_pred):
            return K.mean(y_true*y_pred)
        D_two_class.compile(loss=wasserstein_loss,
                      optimizer='adam',
                      metrics=['accuracy'])

        GD = Sequential()
        GD.add(G)
        D.trainable = False
        GD.add(D_two_class)
        #GD.compile(loss='categorical_crossentropy',
        #              optimizer='adam',
        #              metrics=['accuracy'])
            
        #GD.compile(loss=neg_loss_unlabeled,
        #              optimizer='adam',
        #              metrics=['accuracy'])
        #use with regular D
        
        def neg_wasserstein_loss(y_true,y_pred):
            return -K.mean(y_true*y_pred)
        def neg_binary_cross_entropy(y_true,y_pred):
            return -K.binary_crossentropy(y_true, y_pred)
        GD.compile(loss=neg_binary_cross_entropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        #use with regular D_two_class

        """if first:
            plot(GD, show_shapes=True,to_file='../models/GAN_linked.png')
            GD.summary()
            for layer in D.layers:                    
                layer.trainable = False
            GD.summary()
            first = False
        """
        X_train = X_labeled[train]
        X_train_labeled = X_labeled[train]
        X_train = np.vstack((X_train,X_unlabeled))
        nb_train_example = X_train.shape[0]
        print X_train.shape
        Y_train = Y_categ_labeled[train]
        Y_train_labeled = Y_categ_labeled[train]
        Y_train = np.vstack((Y_train,Y_categ_unlabeled))
        #Y_train[0:int(nb_train_example*0.9)] = [[1,1,0]]*int(nb_train_example*0.9)
        relabeled_Y = Y_train.copy()
        print relabeled_Y.shape
        misclass_counts = np.array([0 for i in range(len(Y_train))])
        unlabeled_indices = []
        labeled_indices = []
        for i in range(X_train.shape[0]):
            if np.allclose(relabeled_Y[i], [0.499,0.499,0.002]):
                unlabeled_indices.append(i)  
            else:
                labeled_indices.append(i)   
        labeled_indices_atstart = list(labeled_indices)
        # Fit the model
        #model.fit(X[train], Y[train], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
        for epoch in range(nb_epoch):
            a1,a2,a3,l3 = ([],[],[],[])
            shuffled = np.random.permutation([i for i in range(X_train_labeled.shape[0])])
            #shuffled = np.random.permutation([i for i in range(X_train_labeled.shape[0])])
            nb_batches = int(np.ceil(X_train_labeled.shape[0]/batch_size))
            shuffled_bigset = np.random.permutation([i for i in range(X_train.shape[0])])
            batch_size_bigset = X_train.shape[0]/nb_batches
            for batch in range(nb_batches):
                #if batch % 10 == 0:
                #    print("%d / %d" % (batch,nb_batches))
                D.trainable = True
                for layer in D.layers:
                    layer.trainable=True
                #loss1,acc1 = D.train_on_batch(X_train[shuffled[batch*batch_size:(batch+1)*batch_size]],relabeled_Y[shuffled[batch*batch_size:(batch+1)*batch_size]])
                loss1,acc1 = D.train_on_batch(X_train_labeled[shuffled[batch*batch_size:(batch+1)*batch_size]],Y_train_labeled[shuffled[batch*batch_size:(batch+1)*batch_size]])
                loss1,acc1 = D_two_class.train_on_batch(X_train[shuffled_bigset[batch*batch_size_bigset:(batch+1)*batch_size_bigset]],[[1]]*batch_size_bigset)
                noise = np.random.normal(0,1,(batch_size,g_input_length,1,1))
                generated = G.predict(noise)
                loss2,acc2 = D.train_on_batch(generated,[[0,0,1]]*batch_size)
                #for l in D.layers:
                #    weights = np.array(l.get_weights())
                #    #weights = [np.clip(w, -10, 10) for w in weights]
                #    new_weights = np.clip(weights,-10,10)                    
                #    l.set_weights(new_weights)
                D.trainable = False
                for layer in D.layers:
                    layer.trainable = False
                #for i in range(3):
                noise = np.random.normal(0,1,(batch_size_bigset,g_input_length,1,1))
                #loss3,acc3 = GD.train_on_batch(noise,[[0,0,1]]*batch_size_bigset)
                loss3,acc3 = GD.train_on_batch(noise,[[0]]*batch_size_bigset)
                #for l in G.layers:
                #    weights = np.array(l.get_weights())
                #    #weights = [np.clip(w, -0.08, 0.08) for w in weights]
                #    new_weights = np.clip(weights,-0.08,0.08)                    
                #    l.set_weights(new_weights)                
                a1.append(acc1)
                a2.append(acc2)
                a3.append(acc3)
                l3.append(loss3)
            
            #count misclassified
            #print D.predict(X_train)
            y_pred_raw = D.predict(X_train)
            y_pred_all = []
            for i in range(X_train.shape[0]):
                to_add = [0,0,0]
                to_add[np.argmax(y_pred_raw[i])] = 1
                y_pred_all.append(to_add)
            y_pred_all = np.array(y_pred_all)

            X_train_labeled = X_train[labeled_indices]
            y_pred_raw_lab = D.predict(X_train_labeled)
            y_pred_labeled = []
            for i in range(len(labeled_indices)):
                to_add = [0,0,0]
                to_add[np.argmax(y_pred_raw_lab[i])] = 1
                y_pred_labeled.append(to_add)
            y_pred_labeled = np.array(y_pred_labeled)        

            X_train_labeled_atstart = X_train[labeled_indices_atstart]
            y_pred_raw_lab_atstart = D.predict(X_train_labeled_atstart)
            y_pred_labeled_atstart = []
            for i in range(len(labeled_indices_atstart)):
                to_add = [0,0,0]
                to_add[np.argmax(y_pred_raw_lab_atstart[i])] = 1
                y_pred_labeled_atstart.append(to_add)
            y_pred_labeled_atstart = np.array(y_pred_labeled_atstart)

            print("epoch: {}".format(epoch))
            #misclass_count_labeled = np.sum( np.max(D.predict(X_train_labeled)-Y_train_labeled,axis=1).astype(int) )
            misclass_count_labeled = np.sum( np.max(y_pred_labeled-Y_train[labeled_indices],axis=1).astype(int) )
            #print "shape",np.max(D.predict(X_train_nounlab)-Y_train_nounlab,axis=1).shape
            misclass_count_all = np.sum( np.max(y_pred_all-Y_train,axis=1).astype(int) )
            print y_pred_all.shape,Y_categ_labeled[train].shape
            misclass_counts += np.max( y_pred_all - Y_train,axis=1).astype(int)
            misclass_count_labeled_atstart = np.sum( np.max(y_pred_labeled_atstart-Y_train[labeled_indices_atstart],axis=1).astype(int) )
            print("misclass_count_labeled: "+str(misclass_count_labeled))
            print("misclassification rate labeled: {} / {} = {}".format(misclass_count_labeled, len(labeled_indices), misclass_count_labeled*1.0 / len(labeled_indices) ))      
            print("misclass_count_all: {}".format(misclass_count_all)) 
            print("misclassification rate all: {} / {} = {}".format(misclass_count_all, nb_train_example, misclass_count_all*1.0/nb_train_example))
            print("misclass_count_labeledatstart: {}".format(misclass_count_labeled_atstart)) 
            print("misclassification rate labeledatstart: {} / {} = {}".format(misclass_count_labeled_atstart, len(labeled_indices_atstart), misclass_count_labeled_atstart*1.0/len(labeled_indices_atstart)))

            if epoch in [20,25]:
                unlabeled_indices = []
                labeled_indices = []
                #relabeled_Y = np.array([Y_train[i] if misclass_count[i]<0.5*epoch else np.array([0.5,0.5,0.0]) for i in range(nb_train_example)])
                for i in range(Y_train.shape[0]):
                    if misclass_counts[i]<0.9*epoch:
                        relabeled_Y[i] = Y_train[i]
                    else:
                        relabeled_Y[i] = [0.499,0.499,0.002]
                    if np.allclose(relabeled_Y[i],[0.499,0.499,0.002]):
                        unlabeled_indices.append(i)
                    else:
                        labeled_indices.append(i)
            X_train_labeled = X_train[labeled_indices]
            Y_train_labeled = Y_train[labeled_indices] 
            print("Epoch: %d SamplesAcc: %f NoiseAcc: %f GeneratorAccLoss: %f %f" % (epoch,np.mean(a1),np.mean(a2),np.mean(a3),np.mean(l3)))
        # evaluate the model
        scores = D.evaluate(X_labeled[test], Y_categ_labeled[test], verbose=1)
        print("%s: %.2f%%" % (D.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        if scores[1] > best_acc:
            D.save_weights(best_weights_filepath, overwrite=True)
        G.save_weights("Generator_weights_"+str(split)+".hdf5", overwrite=True)
        split += 1

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def test_on_file(weights_file,test_file,threshold=0.5,verbose=False):
    #print(weights_file)
    #print(test_file)
    X, Y = load_data(fname=test_file) 
    input_shape = (X.shape[1], 1, 1)
    # convert into 4D tensor For 2D data (e.g. image), "tf" assumes (rows, cols, channels) 
    X = X.reshape(X.shape[0], X.shape[1], 1, 1)
    #print('X shape:', X.shape)

    nb_example = X.shape[0]
    nb_features = X.shape[1]
    Y_categ = []
    for i in range(nb_example):
        if Y[i]>0.5:
            Y_categ.append([0.0,1.0,0.0])
        else:
            Y_categ.append([1.0,0.0,0.0])
    Y_categ=np.array(Y_categ)

    D = discriminator_model(input_shape,weights_file)
    if verbose:
        scores = D.evaluate(X, Y_categ, verbose=1)
        print("%s: %.2f%%" % (D.metrics_names[1], scores[1]*100))

    predictions = D.predict(X)
    predictions = predictions[:,0:2] / np.sum(predictions[:,0:2],axis=1).reshape(-1,1)    
    #print(predictions)
    true_pos,false_pos,true_neg,false_neg = (0,0,0,0)
    for i in range(nb_example):
        """if np.isclose(Y[i],1) and predictions[i][1]>0.5:
            true_pos += 1
        elif np.isclose(Y[i],0) and predictions[i][1]>0.5:
            false_pos += 1
        elif np.isclose(Y[i],0) and predictions[i][0]>0.5:
            true_neg += 1
        elif np.isclose(Y[i],1) and predictions[i][0]>0.5:
            false_neg += 1"""

        if np.isclose(Y[i],1) and predictions[i][1]>threshold:
            true_pos += 1
        elif np.isclose(Y[i],0) and predictions[i][1]>threshold:
            false_pos += 1
        elif np.isclose(Y[i],0) and predictions[i][0]>1-threshold:
            true_neg += 1
        elif np.isclose(Y[i],1) and predictions[i][0]>1-threshold:
            false_neg += 1
    if verbose:
        print("nb_example: {}".format(nb_example))
        print([[true_neg,false_pos],[false_neg,true_pos]])
        print("row 1 neg_act row 2 pos_act")
        print("col 1 neg_pred col 2 pos_pred")
    reread = pd.read_csv(test_file)    
    nb_patients = len(set(reread['INFO'].values))
    return {'true_neg':true_neg,'false_pos':false_pos,'false_neg':false_neg,
            'true_pos':true_pos,'total':nb_example,'nb_patients':nb_patients}

def predict_to_file(weights_file,test_file,outfile):
    #print(weights_file)
    #print(test_file)
    X, Y = load_data(fname=test_file) 
    input_shape = (X.shape[1], 1, 1)
    # convert into 4D tensor For 2D data (e.g. image), "tf" assumes (rows, cols, channels) 
    X = X.reshape(X.shape[0], X.shape[1], 1, 1)
    #print('X shape:', X.shape)

    nb_example = X.shape[0]
    nb_features = X.shape[1]
    Y_categ = []
    for i in range(nb_example):
        if Y[i]>0.5:
            Y_categ.append([0.0,1.0,0.0])
        else:
            Y_categ.append([1.0,0.0,0.0])
    Y_categ=np.array(Y_categ)

    D = discriminator_model(input_shape,weights_file)
    predictions = D.predict(X)
    predictions = predictions[:,1:2] / np.sum(predictions[:,0:2],axis=1).reshape(-1,1)
    predictions = predictions.T[0]
    input_data = pd.read_csv(test_file)
    input_data['GAN_prob'] = pd.Series(predictions)
    input_data.to_csv(outfile,sep=",")
    
if __name__=="__main__":
    if sys.argv[1] == "training":
        training(sys.argv[0:])
    elif sys.argv[1] == "test":
        test_on_file(sys.argv[2],sys.argv[3],verbose=True)
    elif sys.argv[1] == "predict_to_file":
        predict_to_file(sys.argv[2],sys.argv[3],sys.argv[4])
