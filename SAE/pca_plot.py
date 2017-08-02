from sklearn.decomposition import PCA
import data_import
import matplotlib.pyplot as plt

data,df = data_import.import_data("~/Dropbox/missense_pred/data/Ben/input_data.HIS.csv",test = 0)

X_train = data[0]['X_train']
X_test = data[0]['X_test']
y_train = data[1]['y_train']
y_test = data[1]['y_test']

pca = PCA(n_components = 2, svd_solver = 'full')
pca.fit(X_test)
print(pca.explained_variance_ratio_)
trans_X = pca.transform(X_test)
'''
for i,item in enumerate(y_test):
    if item == 1:
        plt.scatter(trans_X[i,0],trans_X[i,1],color = 'red')
    else:
        plt.scatter(trans_X[i,0],trans_X[i,1],color = 'blue')

plt.show()
'''

for i,item in enumerate(y_test):
    if item == 1:
        cut = i

plt.scatter(trans_X[0:cut,0],trans_X[0:cut,1],color = 'red')
plt.scatter(trans_X[cut:,0],trans_X[cut:,1],color='blue')
plt.show()
