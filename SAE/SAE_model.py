from keras.layers import Dense, Dropout
from keras import regularizers
from keras.models import Model, Sequential
import data_import
import pickle as pkl
import json
from keras.utils import plot_model
#data = data_import.import_data("~/Dropbox/missense_pred/data/Ben/inputs/input_data.HIS.csv",test = 1)

data = data_import.import_data("~/Dropbox/missense_pred/data/Ben/input_data.HS.csv",test = 1)

X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']

Model1 = Sequential()
Model1.add(Dense(40,input_shape = (46,),activation = 'relu',name = 'inter1'))
Model1.add(Dense(46,activation = 'relu'))

Model1.compile(optimizer = 'adam',loss = 'mean_squared_error')
Model1.fit(X_train,X_train,epochs = 100,batch_size = 1000,validation_data = (X_test,X_test))

weight_1 = Model1.get_weights()[0]
with open("weight_1",'wb') as f:
    pkl.dump(weight_1,f)

middle1 = Model(inputs = Model1.input,outputs = Model1.get_layer('inter1').output)
middle1_out = middle1.predict(X_train)
middle1_out_test = middle1.predict(X_test)



Model2 = Sequential()
Model2.add(Dense(30,input_shape = (40,),activation = 'relu',name = 'inter2'))
Model2.add(Dense(40,activation = 'relu'))
Model2.compile(optimizer = 'adam',loss = 'mean_squared_error')
Model2.fit(middle1_out,middle1_out,epochs = 100,batch_size = 1000, validation_data = (middle1_out_test,middle1_out_test))

weight_2 = Model2.get_weights()[0]
with open("weight_2",'wb') as f:
    pkl.dump(weight_2,f)


middle2 = Model(inputs = Model2.input, outputs = Model2.get_layer('inter2').output)
middle2_out = middle2.predict(middle1_out)
middle2_out_test = middle2.predict(middle1_out_test)


Model3 = Sequential()
Model3.add(Dense(20,input_shape = (30,),activation = 'relu',name = 'inter3'))
Model3.add(Dense(30,activation = 'relu'))
Model3.compile(optimizer = 'adam',loss = 'mean_squared_error')
Model3.fit(middle2_out,middle2_out,epochs = 100,batch_size = 1000, validation_data = (middle2_out_test,middle2_out_test))

weight_3 = Model3.get_weights()[0]
with open("weight_3",'wb') as f:
    pkl.dump(weight_3,f)

# initializer for pre-trained weights
def my_init1(shape, dtype = None):
    return weight_1

def my_init2(shape,dtype = None):
    return weight_2

def my_init3(shape,dtype = None):
    return weight_3


mlp = Sequential()
mlp.add(Dropout(0,input_shape=(46,)))
mlp.add(Dense(40,activation = 'relu',kernel_initializer = my_init1,activity_regularizer=regularizers.l1(10e-7)))
mlp.add(Dropout(0.2))
mlp.add(Dense(30,activation ='relu',kernel_initializer = my_init2,activity_regularizer=regularizers.l1(10e-7)))
mlp.add(Dropout(0.2))
mlp.add(Dense(20,activation = 'relu',kernel_initializer = my_init3,activity_regularizer=regularizers.l1(10e-7)))
mlp.add(Dense(1,activation = 'sigmoid'))
mlp.compile(loss = 'binary_crossentropy',
		optimizer = 'adam',
		metrics = ['accuracy'])
'''
data = data_import.import_data("~/Dropbox/missense_pred/data/john/HIS_metaSVM_addtest1.anno.rare.reformat.csv",test = 1)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
'''

mlp.fit(X_train,y_train,epochs = 100,batch_size = 200)
SAE_struct = mlp.to_json()
with open("SAE_arc.txt",'w') as f:
    json.dump(SAE_struct,f)
mlp.save_weights('SAE_weights.h5')
plot_model(mlp, to_file='mlp.png')
#building the autoencoder
'''
encoding_dim = 30
input_data = Input(shape = (40,))
encoded = Dense(encoding_dim,activation = 'relu')(input_data)
decoded = Dense(40,activation = 'relu')(encoded)

autoencoder = Model(input_data,decoded)

encoder = Model(input_data,encoded)

encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer = 'adam',loss = 'mean_squared_error')

autoencoder.fit(X_train,X_train,epochs = 100,batch_size = 1000,validation_data = (X_test,X_test))
'''




