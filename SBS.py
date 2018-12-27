from sklearn.metrics import accuracy_score
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone


class SBS(object):


    def __init__(self, estimator, k_features,scoring=accuracy_score, test_size=0.25, random_state=1):
        #Scoring is used to evaluate the perfromance for a feature
        self.scoring = scoring
        self.estimator = clone(estimator)
        #k_features is how many features we want to return
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(cls,x,y):
        x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=cls.test_size, random_state=cls.random_state)

        dim = x_train.shape[1]
        cls.indices_ = tuple(range(dim))
        cls.subsets_ = [cls.indices_]

        score = cls._calc_score(x_train, y_train,x_test,y_test,cls.indices_)

        cls.scores_ = [score]
        w = 1
        v = 1
        while dim > cls.k_features:
            scores_ = []
            subsets = []

            for p in combinations(cls.indices_, r=dim-1):
                score = cls._calc_score(x_train, y_train, x_test, y_test, p)

                scores_.append(score)
                subsets.append(p)

            #np.argmax returns the index of the largest value
            best = np.argmax(scores_)
            cls.indices_ = subsets[best]
            cls.subsets_.append(cls.indices_)
            dim -= 1

            cls.scores_.append(scores_[best])
        cls.k_score_ = cls.scores_[-1]

        return cls

    def transform(cls,x):
        return x[:,cls.indices_]

    def _calc_score(cls, x_train, y_train, x_test, y_test, indices):
        cls.estimator.fit(x_train[:,indices], y_train)
        y_pred = cls.estimator.predict(x_test[:,indices])
        score = cls.scoring(y_test, y_pred)
        return score
