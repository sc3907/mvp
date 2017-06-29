from keras.models import Sequential
import data_import
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
model = Sequential()
model.add(Dense(30,input_shape = (40,),activation = 'relu'))
model.add(Dense(25,activation ='relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.load_weights("SAE_weights.h5")

def plot_roc(df, y_true, label):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    algos = ["SAE_prob",'MetaSVM_rankscore', 'MetaLR_rankscore', 'M-CAP_rankscore',
             'DANN_rankscore','CADD_phred', 'Eigen-phred','Polyphen2_HVAR_rankscore',
             'MutationTaster_converted_rankscore', 'FATHMM_converted_rankscore',
             'fathmm-MKL_coding_rankscore']
    for algo in algos:
        index = df[algo]!= -1
        y_score = df.ix[index][algo].values
        y_true_nomissing = y_true[index]

        fpr[algo], tpr[algo], _ = metrics.roc_curve(y_true_nomissing, y_score)
        roc_auc[algo] = metrics.auc(fpr[algo], tpr[algo])
    # jump comes from missing value

    plt.figure(figsize = (10,10))
    lw = 2
    for algo in algos:
        plt.plot(fpr[algo], tpr[algo], lw=lw,
                 label='{} ROC curve (area = {:.2f})'.format(algo, roc_auc[algo]))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    total = len(y_true)
    pos = sum(y_true)
    neg = total - pos

    plt.title('Receiver operating characteristic of {}: {} positive, {} negative'.format(label, pos, neg))
    plt.legend(loc="lower right", fontsize = 'medium')
    plt.show()

fname = "/Users/bindy/Dropbox/missense_pred/data/john/HS_metaSVM_addtest2.anno.rare.reformat.csv"
fname2 = "/Users/bindy/Dropbox/missense_pred/data/cancer_hotspots/cancer.HS2.reformat.cnn.csv"
data = data_import.import_data(fname)
data2 = data_import.import_data(fname)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
df = pd.read_csv(fname)
y_true = df.pop('target')
y_score = model.predict_proba(X_test,batch_size = 20, verbose = 1)
df = df.assign(SAE_prob = y_score)
plot_roc(df,y_true,label = "HS_metaSVM_addtest2")

