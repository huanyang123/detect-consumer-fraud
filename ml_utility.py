'''
----------------------------------------------------------------------------------------------------
This is a general utility function to perform various tasks in using ML algorithms
----------------------------------------------------------------------------------------------------
By Huanwang Shawn Yang  (2017-06-12)
'''

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



#==========================
def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)


#==========================
def visulization_reg(df):
    '''also check how target correlated with each feature 
    '''
    return
#==========================
def visulization(df):
    '''see if data has problems
    '''
    
# separate variables into nemerical and categorical data data for checking
    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])

    '''
    sns.distplot(df['target']) #plot
    print "The skewness of target is {}".format(df['target'].skew())

    sm.qqplot(df['target'], stats.t, fit=True, line='45') #show if normal dist
    plt.show()
    '''

    #sns.distplot(np.log(df['target']))  #more skewed!
    #plt.show()
    #GrLivArea variable
    #sns.jointplot(x=num['reso'], y=num['target'])
    #plt.show()

    '''
    corr=num.corr()
    sns.heatmap(corr, cmap='RdYlGn_r', linewidths=0.5) #correlation maps
    print corr.sort_values(num.columns.values[0]) #
    plt.show()
    '''

    #print num['reso'].unique()
    #pivot = num.pivot_table(index='reso', values='target', aggfunc=np.median)
    #print pivot
    #pivot.plot(kind='bar', color='red')
    #plt.show()


    #create numeric plots
    num_col = [col for col in num.columns if num.dtypes[col] != 'object']
    nd = pd.melt(num, value_vars = num_col)
    n1 = sns.FacetGrid (nd, col='variable', col_wrap=2, sharex=False, sharey = False)
    n1 = n1.map(sns.boxplot, 'value')
    n1
    n2 = sns.FacetGrid (nd, col='variable', col_wrap=2, sharex=False, sharey = False)
    n2 = n2.map(sns.distplot, 'value')
    n2
    
    '''
    #create categorical plots
    cat = [col for col in df.columns if df.dtypes[col] == 'object']
  #  p = pd.melt(num, id_vars='target', value_vars=cat)
    p = pd.melt(num,  value_vars=cat)
    g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
   # g = g.map(boxplot, 'value','target')
    g = g.map(boxplot, 'value')
    g
    # protein structure have high quality
    cat_pivot = df.pivot_table(index='type', values='target', aggfunc=np.median)
    print cat_pivot
    cat_pivot.plot(kind='bar',color='red')
    print df.describe(), df['type'].value_counts()
    '''
#    sns.pairplot(df, x_vars=['reso','rfree','B'], y_vars='target', size=7, aspect=0.7, kind='reg')

#========================================
def plot_feature(select):
    '''plot feature as index
    '''
    mask=select.get_support()
    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
    plt.xlabel('Index of Features')
    
#========================================
def transf(select, X_train, y_train, X_test, y_test):
    '''
    '''
    from sklearn.linear_model import LogisticRegression

    select.fit(X_train, y_train)
    X_train_s=select.transform(X_train)
    X_test_s=select.transform(X_test)
    score=LogisticRegression().fit(X_train_s, y_train).score(X_test_s, y_test)
    print 'The shape of Xtrain = ', X_train.shape
    print 'The shape of Xtrain_select = ', X_train_s.shape
    print 'The socre of LogisticRegression = %.4f'  %score

    return X_train_s, X_test_s
#========================================
def feature_selection(X_train, X_test, y_train, y_test, ftype, nfeature):
    '''select the best feature to reduce noise, overfit
        X & y are the train, test for feature and target
        ftype: feature type;  nfeature: number of features


Feature selection (only keep the important ones for good performance)::
Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct 
a model and choose either the best or worst performing feature, setting the feature 
aside and then repeating the process with the rest of the features. This process 
is applied until all features in the dataset are exhausted. The goal of RFE is to 
select features by recursively considering smaller and smaller sets of features.

It enables the machine learning algorithm to train faster.
It reduces the complexity of a model and makes it easier to interpret.
It improves the accuracy of a model if the right subset is chosen.
It reduces overfitting.

    '''

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor

#Univariate feature selection works by selecting the best features 
# based on univariate statistical tests
    if ftype=='CHI2': 
        select = SelectKBest(score_func=chi2, k=nfeature)
        X_train_s, X_test_s = transf(select, X_train, y_train, X_test, y_test)
        return X_train_s, X_test_s 
        '''
        select = model.fit(X_train, y_train)
        features = select.transform(X_train)

        print('testing feature score=', fit.scores_)
        print('features=',features[:5,])
        return features
        '''

    elif ftype=='RFE': #  Recursive Feature Elimination (RFE)
        model = RandomForestRegressor(n_jobs=-1, n_estimators=100)
        select = RFE(model, nfeature)
        X_train_s, X_test_s = transf(select, X_train, y_train, X_test, y_test)
        return X_train_s, X_test_s 

        '''
    #    print '\nclf.support_=', clf.support_
        X_col=X.columns

        features=[]  #get the best features
        for i, x in enumerate(clf.support_):
            if x:
                features.append(X_col[i])
        print  'Selected Features=', features

        return features
        '''

    elif ftype=='model':  #model based method
        model=RandomForestRegressor(n_estimators=100, random_state=42)
        select=SelectFromModel(model, threshold=None)
        X_train_s, X_test_s = transf(select, X_train, y_train, X_test, y_test)

        return X_train_s, X_test_s
    


#

