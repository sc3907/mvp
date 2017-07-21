from keras.models import Sequential
import data_import
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
def convert2binary(df, col, threshold):
    '''take a dataframe, col to compare, threshold, return the binary vector
        convert to more elegent lambda function way
        http://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    '''
    values = np.array(df[col].values)
    index = values >= threshold
    values[index] = 1
    values[~index] = 0
    return values

s_het = np.load('/Users/bindy/Dropbox/missense_pred/data/gene/s_het.npy').item()
prec = np.load('/Users/bindy/Dropbox/missense_pred/data/gene/prec.npy').item()
pli = np.load('/Users/bindy/Dropbox/missense_pred/data/gene/pli.npy').item()
lofz = np.load('/Users/bindy/Dropbox/missense_pred/data/gene/lofz.npy').item()

HIS_gene = set(gene for gene, pli_score in pli.items() if pli_score < 0.5)
HS_gene = set(gene for gene, pli_score in pli.items() if pli_score >= 0.5)
prec_5 = set(gene for gene, pli_score in prec.items() if pli_score >0.5)
lofz3 = set(gene for gene, score in lofz.items() if score >= 3)

geneset = HS_gene #& lofz3

model = Sequential()
model.add(Dense(30,input_shape = (39,),activation = 'relu'))
model.add(Dense(25,activation ='relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.load_weights("SAE_weights.h5")

# ASD case
fpath = "~/Dropbox/missense_pred/data/Ben/ASDonly.case.anno.rare.HS.reformat.GCcorrected.cnn.csv"
#fpath = '~/Dropbox/missense_pred/data/john/HIS_case.anno.rare.reformat.csv'
data = data_import.import_data(fpath)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
df = pd.read_csv(fpath)
y_true = df.pop('target')
y_score = model.predict_proba(X_test,batch_size = 20, verbose = 1)
CASE = df.assign(SAE_prob = y_score)


#CHD case
fpath = "~/Dropbox/missense_pred/data/Ben/chd_yale.anno.rare.HS.reformat.GCcorrected.cnn.csv"
data = data_import.import_data(fpath)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
df = pd.read_csv(fpath)
y_true = df.pop('target')
y_score = model.predict_proba(X_test,batch_size = 20, verbose = 1)
CHD = df.assign(SAE_prob = y_score)


#ssc_yale case

fpath = "~/Dropbox/missense_pred/data/Ben/ssc_yale.anno.rare.HS.reformat.GCcorrected.cnn.csv"
data = data_import.import_data(fpath)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
df = pd.read_csv(fpath)
y_true = df.pop('target')
y_score = model.predict_proba(X_test,batch_size = 20, verbose = 1)
ssc_yale = df.assign(SAE_prob = y_score)

#DDD df
fpath = "~/Dropbox/missense_pred/data/Ben/DDD_new_0.2.anno.rare.HS.reformat.GCcorrected.cnn.csv"
data = data_import.import_data(fpath)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
df = pd.read_csv(fpath)
y_true = df.pop('target')
y_score = model.predict_proba(X_test,batch_size = 20, verbose = 1)
df= df.assign(disease = 'DDD_new')
DDD = df.assign(SAE_prob = y_score)


#control df
fpath = "~/Dropbox/missense_pred/data/Ben/control_1911.anno.rare.HS.reformat.GCcorrected.cnn.csv"
data = data_import.import_data(fpath)
X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']
df = pd.read_csv(fpath)
y_true = df.pop('target')
y_score = model.predict_proba(X_test,batch_size = 20, verbose = 1)
control = df.assign(SAE_prob = y_score)


def display_enrichment(case_info, control_info, case_disease, geneset, sort_key='Col'):

    df_case, case_disease, disease_size = case_info
    df_control, control_disease, disease_size = control_info
    case_size = 0.0
    for disease in case_disease:
        if disease in disease_size:
            case_size += disease_size[disease]
    control_size = 0.0
    for disease in control_disease:
        control_size += disease_size[disease]
    print ('case size:{} control size:{}'.format(case_size, control_size))

    # add PDmis columns
    df_case['PDmis'] = np.where((df_case['CADD_phred'] >= 15) & (df_case['Polyphen2_HDIV_rankscore'] >= 0.52875) , 1, 0)
    df_control['PDmis'] = np.where((df_control['CADD_phred'] >= 15) & (df_control['Polyphen2_HDIV_rankscore'] >= 0.52875) , 1, 0)
    # 'M-CAP_rankscore', 0.4815 is 0.025 cutoff 0.642 is 0.05 cutoff
    col_dict={'cadd15':('CADD_phred', 15), 'cadd20':('CADD_phred', 20),
              'cadd25':('CADD_phred', 25), 'cadd30':('CADD_phred', 30),
              'eigen_pred10':('Eigen-phred', 10), 'eigen_pred15':('Eigen-phred', 15),
              'eigen_pc_pred10':('Eigen-PC-phred', 10),
              'MetaSVM>0':('MetaSVM_rankscore', 0.82271),'MetaLR>0':('MetaLR_rankscore', 0.81122),
              'M_CAP>0.025':('M-CAP_rankscore', 0.4815), 'M_CAP>0.05':('M-CAP_rankscore', 0.642),
              'PP2-HVAR':('Polyphen2_HVAR_rankscore', 0.6280),'FATHMM':('FATHMM_converted_rankscore', 0.8235),
              'all_missense':('SAE_prob', 0.0),'SAE_0.05':('SAE_prob', 0.05),
              'SAE_0.1':('SAE_prob', 0.1), 'SAE_0.2':('SAE_prob', 0.2),
              'SAE_0.3':('SAE_prob', 0.3), 'SAE_0.4':('SAE_prob', 0.4),
              'SAE_0.5':('SAE_prob', 0.5), 'SAE_0.6':('SAE_prob', 0.6),
              'SAE_0.7':('SAE_prob', 0.7), 'SAE_0.8':('SAE_prob', 0.8),
              'SAE_best_0.56':('SAE_prob', 0.56),'REVEL_0.5':('REVEL',0.5),
              'REVEL_0.6':('REVEL',0.6),'REVEL_0.7':('REVEL',0.7),'REVEL_0.8':('REVEL',0.8),
              'REVEL_0.9':('REVEL',0.9),'MPC_1':('MPC',1),'MPC_2':('MPC',2)}


    infos = []
    for col_name, (col, threshold) in col_dict.items():
        #print(df_case)
        #print(convert2binary(df_case,col,threshold))
        #case_count = np.sum(convert2binary(df_case, col, threshold))
        #control_count = np.sum(convert2binary(df_control, col, threshold))
        case_count = np.sum([1 if i>threshold else 0 for i in df_case[col] ])
        control_count = np.sum([1 if i>threshold else 0 for i in df_control[col]])
        total_counts = case_count + control_count
        #control_count = max(control_count, 1)
        print(case_count,case_size,control_count,control_size)
        enrich = float(case_count) / case_size / (float(control_count) / control_size)
        pvalue = scipy.stats.binom_test(case_count, total_counts,
                                            case_size / (case_size + control_size))
        risk_gene = case_count * (enrich - 1) / enrich
        #enrich = max(enrich, 1)
        tpr = (enrich - 1) / enrich

        #exp = mutation_bkgrd.expectation(geneset, col_name) * case_size
        #exp_enr = case_count / exp
        #exp_risk_gene = case_count * (exp_enr - 1) / exp_enr
        #exp_tpr = (exp_enr - 1) / exp_enr


        infos.append([col_name, case_count, control_count,
                      enrich, pvalue, risk_gene, tpr])


    labels = ['Col', 'Case', 'Control', 'enrich', 'pvalue', '# risk gene', 'true positive rate']
    df = pd.DataFrame(infos,columns=labels)
    df = df.sort_values(by=sort_key, ascending=True)
    print(df)
    plot_rate_vs_riskvariants(df, title=','.join(case_disease))
    return df

def plot_rate_vs_riskvariants(df, title):
    x = list(df['true positive rate'])
    y = list(df['# risk gene'])

#     x = list(df['exp_tpr'])
#     y = list(df['exp_risk_gene'])
    methods = list(df['Col'])
    fig, ax = plt.subplots(figsize = (15,10))
    ax.scatter(x, y, s=100)
    for i, txt in enumerate(methods):
        if 'SAE' in txt:
            color = 'red'
        elif 'all_missense' in txt:
            color = 'blue'
        else:
            color = 'black'
        ax.annotate(txt, (x[i],y[i]), fontsize=15, color=color)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel('true positive rate', fontsize=15)
    ax.set_ylabel('# risk gene', fontsize=15)

    case_variants, control_variants = df.ix[df['Col']=='all_missense'][['Case', 'Control']].values[0]

    ax.set_title('{}: {} variants in cases, {} variants in controls'.format(title,
                                            case_variants, control_variants), fontsize=15)

    plt.show()


disease_size = {'CHD':2779, 'CDH':307,'CDH_MGH':80,
                'ASD':3953, 'EE':264, 'ID':192,
                'DDD_new':4293, 'DDD':1133, 'SSC':1911,
                'CHD_yale':2645, 'SSC_yale':1789 }

control_disease = ['SSC']
df_control = control
index = df_control['disease'].isin(control_disease)
df_control = df_control[index]
control_info = (df_control, control_disease, disease_size)

#plot ASD
case_disease = ['ASD']
df_case = CASE
index =  df_case['disease'].isin(case_disease)
df_case = df_case[index]
case_info = (df_case, case_disease, disease_size)
df = display_enrichment(case_info, control_info, case_disease, geneset)

#plot CHD_yale
case_disease = ['CHD_yale']
df_case = CHD
index =  df_case['disease'].isin(case_disease)
df_case = df_case[index]
case_info = (df_case, case_disease, disease_size)
df = display_enrichment(case_info, control_info, case_disease, geneset)

#plot SSC_yale
case_disease = ['SSC_yale']
df_case = ssc_yale
index =  df_case['disease'].isin(case_disease)
df_case = df_case[index]
case_info = (df_case, case_disease, disease_size)
df = display_enrichment(case_info, control_info, case_disease, geneset)

#plot DDD
case_disease = ['DDD_new']
df_case = DDD
index = df_case['disease'].isin(case_disease)
df_case = df_case[index]
case_info = (df_case,case_disease, disease_size)
df = display_enrichment(case_info,control_info,case_disease, geneset)

