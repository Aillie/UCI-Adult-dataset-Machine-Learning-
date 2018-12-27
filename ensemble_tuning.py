from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import numpy as np
import warnings


def ensemble_tuning(model_list,x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
    vc = VotingClassifier(estimators=model_list, voting='hard')
    vc.fit(x,y)
    y_pred = vc.predict(x_test)
    #y_pred_log = vc.predict_proba(x_test)
    y_pred_t = vc.predict(x_train)
    #y_pred_log_t = vc.predict_proba(x_train)
    print(f'Accuracy score [test]: {accuracy_score(y_test,y_pred)}')
    print(f'Accuracy score [train]: {accuracy_score(y_train, y_pred_t)}')
    #print(f'Log loss [test]: {log_loss(y_test, y_pred_log)}')
    #print(f'Log loss [train]: {log_loss(y_train, y_pred_log_t)}')

def data_fisher():
    x = np.genfromtxt('res_data.csv', delimiter=',')
    y = np.genfromtxt('y.csv', delimiter=',')
    return x,y

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    model_list = [('SGD',SGDClassifier(loss='log', max_iter=50, penalty='elasticnet',l1_ratio=0.35,learning_rate='invscaling', eta0=0.5)),('LR',LogisticRegression(C=26, max_iter=14, penalty='l2', solver='newton-cg')),('DTC',DecisionTreeClassifier(max_depth=6, max_features=7,random_state=1, min_samples_split=9, min_samples_leaf=7))]
    x,y = data_fisher()

    ensemble_tuning(model_list,x,y)
