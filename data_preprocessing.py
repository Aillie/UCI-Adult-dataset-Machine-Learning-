from data import adult_data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from SBS import SBS
import matplotlib.pyplot as plt
import csv


def drop_features():
    data = adult_data.drop(['education', 'fnlwgt', 'relationship'], axis=1)
    return data


def gain():
    x=[]
    for v in adult_data['capital-gain']:
        if v > 14084:
            x.append('High')
        elif v >= 3411 and v < 14084:
            x.append('Medium')
        else:
            x.append('Low')

    adult_data['capital-gain'] = x

def loss():
    x=[]
    for v in adult_data['capital-loss']:
        if v > 1977:
            x.append('High')
        elif v >= 1672 and v < 1977:
            x.append('Medium')
        else:
            x.append('Low')

    adult_data['capital-loss'] = x

def shuffle_data():
    data_copy = adult_data.loc[adult_data['income'] == 2]
    for col in data_copy:
        x = data_copy[col]
        x = x.sample(frac=1, random_state=7)
        x.reset_index(inplace=True, drop=True)
        x = list(x)
        data_copy[col] = x
    return data_copy

def race_change():
    global adult_data
    adult_data['race'].replace(' Amer-Indian-Eskimo', 'other', inplace=True)
    adult_data['race'].replace('  Asian-Pac-Islander', 'other', inplace=True)

def age_cat():
    global adult_data
    x = []
    for age in adult_data['age']:
        if age <= 23:
            x.append('a')
        elif age > 23 and age <=37:
            x.append('b')
        elif age > 37 and age <= 61:
            x.append('c')
        elif age > 61 and age <= 73:
            x.append('d')
        else:
            x.append('e')
    adult_data['age'] = x

def hours_cat():
    global adult_data
    x = []
    for i in adult_data['hours-per-week']:
        if i > 0 and i < 21:
            x.append('a')
        elif i > 20 and i < 41:
            x.append('b')
        elif i > 40 and i < 61:
            x.append('c')
        elif i > 60 and i < 81:
            x.append('d')
        else:
            x.append('e')
    adult_data['hours-per-week'] = x

def cat_data():
    global adult_data, adult_data_cat
    adult_data_cat = adult_data.select_dtypes(include=['object'])
    adult_data = adult_data.select_dtypes(exclude=['object'])

def impute_missing_values():
    global adult_data_cat

    si = SimpleImputer(missing_values=' ?', strategy='most_frequent')
    data = pd.DataFrame(si.fit_transform(adult_data_cat), columns=adult_data_cat.columns)
    return data

def regions():
    global adult_data_cat
    for v in adult_data_cat['native-country']:
        adult_data_cat['native-country'].replace(v,v.lstrip(),inplace=True)

    adult_data_cat['native-country'].replace(to_replace=['Puerto-Rico', 'Dominican-Republic', 'El-Salvador', 'Holand-Netherlands','Outlying-US(Guam-USVI-etc)', 'United-States','Trinadad&Tobago'], value=['Puerto_Rico','Dominican_Republic','El_Salvador','Netherlands','Outlying_US', 'United_States', 'TrinadadTobago'], inplace=True)

    regions = dict(United_States='Central_America', Mexico='South_America', Philippines='South_East_Asia', Germany='Europe', Canada='Central_America', Puerto_Rico='South_America', El_Salvador='South_America', India='South_Asia', Cuba='South_America', England='Europe', Jamaica='South_America', South='Africa', China='East_Asia', Italy='Europe', Dominican_Republic='South_America', Vietnam='South_East_Asia', Guatemala='South_America', Japan='East_Asia', Poland='East_Europe', Columbia='South_America', Taiwan='South_East_Asia', Haiti='South_America', Iran='South_Asia', Portugal='Europe', Nicaragua='South_America', Peru='South_America', Greece='Europe', France='Europe', Ecuador='South_America', Ireland='Europe', Hong='South_East_Asia', Cambodia='South_East_Asia', TrinadadTobago='South_America', Laos='South_East_Asia', Thailand='South_East_Asia', Yugoslavia='East_Europe', Outlying_US='Central_America', Hungary='East_Europe', Honduras='South_America', Scotland='Europe', Netherlands='Europe')

    adult_data_cat['region'] = adult_data_cat['native-country'].map(regions)
    adult_data_cat = adult_data_cat.drop(['native-country'], axis=1)

def gain_loss_encode():
    global adult_data_cat
    mapping = dict(Low=1, Medium=2, High=3)
    for col in adult_data_cat[['capital-gain', 'capital-loss']]:
        adult_data_cat[col] = adult_data_cat[col].map(mapping)

def age_encode():
    global adult_data_cat
    mapping = dict(a=1,b=2,c=3,d=4,e=5)
    adult_data_cat['age'] = adult_data_cat['age'].map(mapping)

def hours_encode():
    global adult_data_cat
    mapping = dict(a=1,b=2,c=3,d=4,e=5)
    adult_data_cat['hours-per-week'] = adult_data_cat['hours-per-week'].map(mapping)

def binarize_encode():
    global adult_data_cat
    adult_data_cat = pd.get_dummies(adult_data_cat)

def drop_cat_feat():
    global adult_data_cat
    drops = ['workclass_ Federal-gov', 'workclass_ Never-worked', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'marital-status_ Married-AF-spouse', 'marital-status_ Married-civ-spouse', 'occupation_ Adm-clerical', 'occupation_ Craft-repair', 'occupation_ Farming-fishing', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Prof-specialty', 'occupation_ Sales', 'occupation_ Tech-support', 'race_ Other', 'race_other', 'sex_ Female', 'sex_ Male', 'region_East_Asia', 'region_East_Europe', 'region_South_America', 'region_South_East_Asia']
    for drop in drops:
        adult_data_cat = adult_data_cat.drop(drop, axis=1)

def rescale_data(data):
    sc = StandardScaler().fit(data)
    res = MinMaxScaler().fit(data)
    data_std = sc.transform(data)
    data_res = res.transform(data)
    return data_std, data_res

def merge_data(data,cat_data):
    x = np.hstack((data,cat_data))
    return x

def data_slpitter(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.23, random_state=0)
    return x_train, x_test, y_train, y_test

def feature_selection_rf(x,y):
    forest = RandomForestClassifier(n_estimators=1000, n_jobs=2, random_state=0)
    forest.fit(x,y)

    labels = adult_data.columns.union(adult_data_cat.columns)
    labels = labels.drop('income')
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    drops = []
    for i in range(x.shape[1]):
        if importances[indices[i]] < 0.005:
            drops.append(labels[indices[i]])
        else:
            print(f'{i}) {labels[indices[i]]}')


def save_data(data, labels):
    for data,name in zip(data,labels):
        np.savetxt(name,data,delimiter=',')

def data_copy():
    data = adult_data.loc[adult_data['income'] == 2].copy()
    data = data.sample(n=9038,random_state=7)
    return data

def sbs_feature_selection(x,y):
    sbs = SBS(estimator=LogisticRegression(), k_features=1)
    sbs.fit(x,y)

    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    #plt.show()

if __name__ == '__main__':
    adult_data = drop_features()
    gain()
    loss()
    race_change()
    age_cat()
    hours_cat()

    more_data = shuffle_data()
    adult_data = adult_data.append(more_data, ignore_index=True)
    more_data = data_copy()
    adult_data = adult_data.append(more_data, ignore_index=True)

    cat_data()
    adult_data_cat = impute_missing_values()
    remove_null_values()
    regions()

    adult_data_cat['education-num'] = adult_data['education-num']
    adult_data = adult_data.drop(['education-num'], axis=1)

    age_encode()
    hours_encode()
    gain_loss_encode()
    binarize_encode()
    drop_cat_feat()

    #x = adult_data.values[:,0]
    y = adult_data.values[:,0]
    x = adult_data_cat.values[:]
    #x = merge_data(x[:,None], x_cat)

    x_std, x_res = rescale_data(x)

    #x = merge_data(x, x_cat)
    #x_std = merge_data(x_std, x_cat)
    #x_res = merge_data(x_res, x_cat)

    x_train, x_test, y_train, y_test = data_slpitter(x,y)
    x_train_std, x_test_std,y_train,y_test = data_slpitter(x_std,y)
    x_train_res,x_test_res, y_train,y_test = data_slpitter(x_res,y)

    #feature_selection_rf(x_train, y_train)
    #sbs_feature_selection(x_train,y_train)

    data = [x_train, x_test, x_train_std,x_test_std,x_train_res,x_test_res,y_train,y_test]
    labels = ['x_train.csv', 'x_test.csv', 'x_train_std.csv', 'x_test_std.csv', 'x_train_res.csv', 'x_test_res.csv', 'y_train.csv', 'y_test.csv']

    data_org = [x,x_std,x_res,y]
    labels_org = ['org_data.csv', 'std_data.csv', 'res_data.csv','y.csv']

    save_data(data,labels)
    save_data(data_org, labels_org)
