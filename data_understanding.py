from data import adult_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def peek():
    print(adult_data.head(20))

def description():
    print('\n',adult_data.describe())

def gain_loss():
    gain_loss = adult_data[['capital-gain', 'capital-loss']]

    gain_loss['capital-gain'] = gain_loss['capital-gain'].replace(0, np.nan)
    gain_loss['capital-loss'] = gain_loss['capital-loss'].replace(0, np.nan)

    print(gain_loss.describe())

    for f in gain_loss:
        x = gain_loss[f].dropna()
        fg,ax = plt.subplots()
        ax.set_title(f)
        ax.boxplot(x)

    plt.show()

def peek_cat():
    cat_data = adult_data.select_dtypes(include=['object'])

    for f in cat_data:
        print(f'\n',cat_data[f].value_counts())

def correlation():
    n_adult_data = adult_data.select_dtypes(exclude=['object'])
    sns.heatmap(n_adult_data.corr(),annot=True)
    plt.show()

def null_values():
    print('\n',adult_data.isnull().sum())

def question_marks():
    for f in adult_data:
        x = []
        for v in adult_data[f]:
            if v == ' ?':
                x.append(v)
            else:
                pass
        print(f'There are {len(x)} ? values in {f}')


def class_instances():
    print('\n',adult_data['income'].value_counts())

def age_peek():
    y = adult_data[['income', 'age']].groupby('age').mean()
    x = y.index.values
    y = y['income'].values

    plt.plot(x,y)
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Mean income')
    plt.show()

def race_peek():
    print(adult_data['race'].value_counts())

def hours_peek():
    y = adult_data[['hours-per-week', 'income']].groupby('hours-per-week').mean()
    x = y.index.values
    y = y.values

    print(np.unique(adult_data['hours-per-week']))

    plt.plot(x,y)
    plt.xlabel('Hours per week')
    plt.ylabel('Income')
    plt.title('Hours per week mean income')
    plt.show()


if __name__ == '__main__':

    #peek()
    #description()
    #gain_loss()
    #peek_cat()
    #correlation()
    #null_values()
    #class_instances()
    #question_marks()
    #age_peek()
    #race_peek()
    hours_peek()

    #print(adult_data['native-country'][14])
