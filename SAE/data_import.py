import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data(path,test_rate = 0.2,test = 0):

    seed = 7
    np.random.seed(seed)
    exclude_cols = {'target', 'CADD_phred', 'xEigen-phred', 'Eigen-PC-phred',
                              'Eigen-PC-raw_rankscore', 'MetaSVM_rankscore',
                              'MetaLR_rankscore', 'M-CAP_rankscore', 'DANN_rankscore',
                              'CADD_raw_rankscore', 'Polyphen2_HVAR_rankscore',
                              'MutationTaster_converted_rankscore',
                              '#chr', 'pos(1-based)',  'hg19_chr', 'hg19_pos(1-based)',
                              'ref', 'alt', 'category',
                              'source', 'INFO', 'disease', 'genename',
                              'pli', 'lofz', 'prec',
                              'x1000Gp3_AF', 'xExAC_AF',
                              's_het', 'xs_het_log', 'xgc_content',
                              'xFATHMM_converted_rankscore', 'xfathmm-MKL_coding_rankscore',
                              'xpreppi_counts', 'xubiquitination','gc_content','BioPlex','Unnamed: 0'
                              ,'gnomad','s_hat','REVEL','RVIS','mis_badness','obs_exp','MPC','cnn_prob'}

    X = pd.read_csv(path)



    y = X.target
    X = X.drop("target",1)
    X = X[X.columns.difference(exclude_cols)]
    print(list(X))
    #print(X['MPC'])
    X = X.values
    y = y.values
    if test == 1:
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_rate, random_state=42)
        #y_train = keras.utils.to_categorical(y_train,2)
        #y_test = keras.utils.to_categorical(y_test,2)
        X = {'X_train':X_train, 'X_test':X_test}
        y = {'y_train':y_train, 'y_test':y_test}
        return (X,y)
    else:
        X = {'X_test' : X,'X_train' : 0}
        y = {'y_test' : y,'y_train' : 0}
        return (X,y)

