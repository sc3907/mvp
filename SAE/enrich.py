from keras.models import Sequential
import data_import
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from MutBkgd import MutationBackground
model = Sequential()
model.add(Dense(30,input_shape = (40,),activation = 'relu'))
model.add(Dense(25,activation ='relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.load_weights("SAE_weights.h5")

fname = '~/Dropbox/data/gene/gene_mutation_rate0520.txt'
mutation_bkgrd = MutationBackground(fname)


