import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data(path,test_rate = 0.2,test = 0):

    seed = 7
    np.random.seed(seed)
    '''
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
    '''
    features = ['MutationAssessor_score_rankscore', 'VEST3_rankscore', 'Polyphen2_HDIV_rankscore', 'SIFT_converted_rankscore', 'PROVEAN_converted_rankscore', 'FATHMM_converted_rankscore', 'GenoCanyon_score_rankscore', 'LRT_converted_rankscore', 'Eigen-PC-raw_rankscore', 'Eigen-phred', 'Eigen-PC-phred', 'phyloP20way_mammalian_rankscore', 'GERP++_RS_rankscore', 'SiPhy_29way_logOdds_rankscore',
'phastCons100way_vertebrate_rankscore', 'fathmm-MKL_coding_rankscore','phyloP100way_vertebrate_rankscore', 'phastCons20way_mammalian_rankscore', 'GM12878_fitCons_score_rankscore', 'HUVEC_fitCons_score_rankscore', 'integrated_fitCons_score_rankscore', 'H1-hESC_fitCons_score_rankscore', 'blosum62', 'pam250', 'SUMO_diff', 'SUMO_score', 'SUMO_cutoff', 'phospho_cutoff', 'phospho_score', 'phospho_diff', 'lofz', 'prec', 'pli','s_het_log', 'secondary_E', 'secondary_H', 'complex_CORUM', 'preppi_counts', 'gnomad', 'ASA', 'secondary_C', 'gc_content', 'interface', 'ubiquitination', 'BioPlex', 'obs_exp']
    X = pd.read_csv(path)



    y = X.target
    df = X
    #X = X.drop("target",1)
    X = X[features]
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

