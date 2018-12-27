from sklearn.model_selection import RandomizedSearchCV, train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings


def model_tuning_random(model,x,y,param_grid):
    model = model()
    grid = RandomizedSearchCV(estimator=model,param_distributions=param_grid,verbose=10,return_train_score=False, n_jobs=5,scoring='neg_log_loss')
    grid.fit(x,y)
    print(grid.best_params_)

def model_tuning_grid(model,x,y,param_grid):
    model = model()
    grid = GridSearchCV(estimator=model,param_grid=param_grid,verbose=10,return_train_score=False, n_jobs=5,scoring='neg_log_loss')
    grid.fit(x,y)
    print(grid.best_params_)

def evaluate_model(model,x,y):
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred_log = model.predict_proba(x_test)
    y_pred_log_t = model.predict_proba(x_train)
    y_pred_t = model.predict(x_train)
    print(f'Accuracy score (test data): {accuracy_score(y_test,y_pred)}')
    print(f'Log loss (test data): {log_loss(y_test,y_pred_log)}')
    print(f'Accuracy score (train data): {accuracy_score(y_train, y_pred_t)}')
    print(f'Log loss (train data): {log_loss(y_train, y_pred_log_t)}')

def data_fisher():
    x = np.genfromtxt('x_train_std.csv',delimiter=',')
    y = np.genfromtxt('y_train.csv',delimiter=',')
    return x,y


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        x,y = data_fisher()

        param_grid = dict()

        model_tuning_grid()

        evaluate_model()
