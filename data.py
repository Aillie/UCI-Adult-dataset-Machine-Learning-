from pandas import read_csv
import numpy as np


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country', 'income']
adult_data = read_csv(url,names=names)

adult_data['income'] = np.where(adult_data['income'] == ' <=50K',1,2)
