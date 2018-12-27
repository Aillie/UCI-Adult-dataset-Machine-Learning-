from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np
import warnings


model_list = [('SVC', SVC())]

dtc = ('DecisionTreeClassifier', DecisionTreeClassifier(criterion='entropy', max_depth=44,random_state=1,max_features=11))
rfc = ('RandomForestClassifier', RandomForestClassifier(n_estimators=1000,n_jobs=5, random_state=1))

def spot_check(models,labels,scoring='accuracy'):
    y = np.genfromtxt(open('y.csv', 'rb'))
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=7)
        for tname in labels:
            x = np.genfromtxt(tname,delimiter=',')
            result = cross_val_score(model,x,y,cv=kfold,scoring=scoring,n_jobs=10)
            print(f'{name} with {tname} data accuracy: {result.mean()}')
        print('\n')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        labels = ['org_data.csv', 'std_data.csv', 'res_data.csv']
        spot_check(model_list,labels)
