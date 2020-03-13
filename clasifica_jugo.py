import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn.model_selection import train_test_split

# Carga datos
data = pd.read_csv('OJ.csv')
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0
data['Target'] = purchasebin
data = data.drop(['Purchase'],axis=1)
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')

# Divide
x = np.asarray(data[predictors])
y = purchasebin
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5)

av_f1_train = []
sigma_f1_train = []
av_f1_test = []
sigma_f1_test = []
av_fi = np.empty((10,(np.shape(x_train)[1])))
max_depth = np.linspace(1,10,10)

for element in max_depth:
    n_points = len(y_train)
    f1_train = []
    f1_test = []
    fi = np.zeros(np.shape(x_train)[1])
    for i in range(0,100):
        indices = np.random.choice(np.arange(n_points), n_points)
        new_x_train = x_train[indices, :]
        new_y_train = y_train[indices]
        new_x_test = x_test[indices, :]
        new_y_test = y_test[indices]
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=element)
        clf.fit(new_x_train, new_y_train)
        
        f1_train.append(sklearn.metrics.f1_score(new_y_train, clf.predict(new_x_train)))
        f1_test.append(sklearn.metrics.f1_score(new_y_test, clf.predict(new_x_test)))
        fi += clf.feature_importances_
    
    av_f1_train.append(np.mean(f1_train))
    sigma_f1_train.append(np.std(f1_train))
    av_f1_test.append(np.mean(f1_test))
    sigma_f1_test.append(np.std(f1_test))
    av_fi[int(element-1),:] = fi/100.
    

plt.figure()
plt.scatter(max_depth,av_f1_train,label = 'train (50%)')
plt.errorbar(max_depth,av_f1_train,yerr = sigma_f1_train)
plt.scatter(max_depth,av_f1_test,label = 'test (50%)')
plt.errorbar(max_depth,av_f1_test,yerr = sigma_f1_test)
plt.legend()
plt.ylabel('Average F1 Score')
plt.xlabel('Max Depth')
plt.savefig('F1_training_test.png')

plt.figure()
for i in range(0,np.shape(x_train)[1]):
    plt.plot(max_depth,av_fi[:,i], label = 'column {}'.format(i))   
plt.legend()
plt.ylabel('Average Feature Importance')
plt.xlabel('Max Depth')
plt.savefig('features.png')
