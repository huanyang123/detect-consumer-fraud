
# coding: utf-8

# # A full analysis for predicting consumer fraud
# By Huanwang Yang 
#  
# The data was obtained from the company (airisDATA)
# 
# # The result:  
# Classification was performed with accuracy of 0.96 and AUC of 0.77.
# 
# 
# # Note: special features to this data set:
# Feature engineering made significant improvement in this case. Great improvement was observed after adding the time difference (between purchase_time and signup_time) as one more feature. Some improvement was observed when the countries were added to the data file. 
# 
# A special python code was written to merge the two files (one with 151113 lines and another with 138847 lines). If using the traditional looping method, it would take hours to finish, since it would perform ~20981586711 operations. Here I sorted two files and split them into 1000 segments. This only takes ~1 min to finish. 
# 
# # =======================================
# 
# # The business problem:
# Company XYZ is an e-commerce site that sells hand-made clothes. You have to build a model that predicts whether a user has a high probability of using the site to perform some illegal activity or not. 
# 
# # The business  objective:
# Build a model to predict whether an activity is fraudulent or not. Explain how different 
# assumptions about the cost of false positives vs false negatives would impact the model.
# 
# # Understand the data below
# 
# 
# Columns from the first file (Fraud_Data.csv): 
# 
# user_id : Id of the user. Unique by user    
# signup_time : the time when the user created her account (GMT time)        
# purchase_time : the time when the user bought the item (GMT time)       
# purchase_value : the cost of the item purchased (USD)          
# device_id : the device id. You can assume that it is unique by device. I.e., transactions    
#             with the same device ID means that the same physical device was used to buy     
# source : user marketing channel: ads, SEO, Direct (i.e. came to the site by directly       
#                                                    typing the site address on the browser)     
# browser : the browser used by the user.     
# sex : user sex: Male/Female       
# age : age of the user         
# user ip_address : user numeric ip address       
# class : this is what we are trying to predict: whether the activity was fraudulent (1) or not (0).     
# 
# 
# Columns from the second file (IpAddress_to_Country.csv):    
# 
# lower_bound_ip_address : the lower bound of the numeric ip address for that country  
# upper_bound_ip_address : the upper bound of the numeric ip address for that country   
# country : the corresponding country. If a user has an ip address whose value is within 
#           the upper and lower bound, then she is based in this country.
# 
# 
# "IpAddress_to_Country" - mapping each numeric ip address to its country. 
# For each country, it gives a range. If the numeric ip address falls within the range, 
# then the ip address belongs to the corresponding country. 
# 
# 
# # Answer to some questions:
# See bottom!
# 

# In[1]:


# import some libraries including the modules (ml_tune_para) that I developed
# Note: The two models must be positioned at the working folder 
import sys
import ml_tune_para as tune
import ml_utility as ut

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from scipy.stats import norm
from sklearn.model_selection import train_test_split

# allow plots to appear in the notebook
get_ipython().magic(u'matplotlib inline')
plt.rcParams['font.size'] = 14


# In[2]:


#convert from files to data frame and do some checkings
df=pd.read_csv('Fraud_Data.csv')
df0=pd.read_csv('IpAddress_to_Country.csv')
df0=df0.sort_values('lower_bound_ip_address')  #for later use
print 'shape of df and df0=', df.shape, df0.shape 
print 'from the first file=\n', df.head(2)
print '\nfrom the second file=\n', df0.head(2)


# In[3]:


# Add countries from pAddress_to_Country.csv to Fraud_Data.csv dataframe
# df and df0 must be sorted!!
def merge_files(df, df0):
    ip0=df0[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']].values

    n=len(ip0)
    bond=[]
    for x in range(0, n, 1000): bond.append(x) #divide for fast calculation
    bond.append(n-1)
    print 'number of segments (for fast calculation) =', len(bond)
    
    cont=[]
    dd=[]
    d1=df['ip_address'].values #already sorted
    nn=0
    for x in d1[:]:
        nn=nn+1
        val=np.nan  #pre assign
        n=0
        for i in range(len(bond)):
            if i==0: continue
            b1, b2 = bond[i-1], bond[i] 
            mini = ip0[b1][0]
            maxi = ip0[b2][0]
            m=b1
            if maxi>= x >=mini : #in the IP boundary
                for y in ip0[b1 : b2]:
                    m=m+1
                    if y[1] >= x >= y[0]:
                        val=ip0[m][2]
                     #   print nn,n, i, m, mini, x, maxi, val
                        n=1
                        break
            if n==1 : break
        cont.append(val)
    df['country']=cont

    return df

df=merge_files(df, df0)


# In[4]:


# When checking the signup_time & purchase_time, it is found that
# the time difference is very correlated with the class (fraud)

#add time difference as one more feature (change to float)
df['signup_time']=pd.to_datetime(df['signup_time'])
df['purchase_time']=pd.to_datetime(df['purchase_time'])
df['dtime'] = (df['purchase_time']-df['signup_time']).astype('timedelta64[h]')


# In[5]:


#Print information to understand the data!

print '\nvalue count for class (target)=\n', df['class'].value_counts()
print '\nvalue count for source=\n', df.source.value_counts()
print '\nvalue count for browser=\n', df.browser.value_counts()
print '\nvalue count for sex=\n', df.sex.value_counts()
print '\nvalue count for age=\n', df.age.value_counts().head()
print '\ngroup by sex (mean)=\n', df.groupby('sex').mean()
print '\ngroup by class (mean)=\n', df.groupby('class').mean()

print '\ndescribe of data file Fraud_Data.csv=\n', df.describe()    
print '\ncheck for missing values (percentage)=\n', df.isnull().sum()/len(df) 

print '\nsorted device_id (class=1)=\n', df.loc[df['class']==1, 'device_id'].value_counts().head()
print '\nsorted device_id (class=0)=\n', df.loc[df['class']==0, 'device_id'].value_counts().head()


# In[6]:


#visulize the data. separate numerical and catergorical 
df_num=df.select_dtypes(include=[np.number])
df_cat=df.select_dtypes(exclude=[np.number])
    
print 'shape of numerical & categorical dataframe=', df_num.shape, df_cat.shape


# In[7]:


#visulize some data
sns.countplot(data=df_num, x='class', palette='hls')
#plt.savefig('count_plot')
plt.show()

#some visualizations
pd.crosstab(df_num.age, df_num['class']).plot(kind='bar')
plt.title('Frequency for Job Title')
#
plt.xlabel('Job')
plt.ylabel('Frequency')
plt.show()


# In[8]:


import ml_utility as ut

print '\nPlot heatmap to show correlations among the features.'
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
pcorr=df_num.astype(float).corr()
sns.heatmap(pcorr,linewidths=0.1,vmax=1.0,
        square=True, cmap=colormap, linecolor='white', annot=True)

#more visulizations 
ut.visulization(df_num)


# In[9]:


#add dummy variables (need number), drop the first one (duplicated)
cat_vars=['sex','browser','source', 'country']
for var in cat_vars:
    cat_list = pd.get_dummies(df[var], prefix=var, drop_first=True)
    df=df.join(cat_list)
    
print 'total number of columns after get_dummies =', len(df.columns.values)
print 'Averaged values for fraudulent case =\n', df[df['class']==1].mean().sort_values().tail() 
print '\nAveraged values for non-fraudulent case =\n', df[df['class']==0].mean().sort_values().tail()


# In[10]:


# drop some features 

feature_drop=['user_id', 'signup_time', 'purchase_time','device_id',  
              'ip_address','sex','browser','source','class', 'country']

feature=[i for i in df.columns.values if i not in feature_drop] 

X_final=df[feature]
y_final=df['class'] 
print 'shape of the final data =', X_final.shape, y_final.shape

#tmpl=df.mean().sort_values().head(150).keys().values.tolist()
#df.drop(tmpl, axis=1, inplace=True)


# In[11]:


#Split the data for training and test set 80/20

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.20)

print 'Training set (%d) and test set (%d).' %(len(X_train), len(X_test))
print 'shape of X_train & y_train =', X_train.shape, y_train.shape


# In[12]:


#Feature selection
import ml_utility as ut

#print X_train.describe()
#X_train, X_test=ut.feature_selection(X_train, X_test, y_train, y_test, 'RFE', 20)
#X_train, X_test=ut.feature_selection(X_train, X_test, y_train, y_test, 'CHI2', 20)
X_train, X_test=ut.feature_selection(X_train, X_test, y_train, y_test, 'model', 7)
print 'After feature selection: shape of X_train & y_train =', X_train.shape, y_train.shape


# In[13]:


#tuning paramters
import ml_tune_para as tune

#tune.auto_tune_classifier(X_train, y_train)
#tune.tune_classifier('RandomForestClassifier', X_train, y_train)
#tune.tune_classifier('LogisticRegression', X_train, y_train)
#tune.tune_classifier('GaussianProcessClassifier', X_train, y_train)
tune.tune_classifier('XGBClassifier', X_train, y_train)
#tune.tune_classifier('GradientBoostingClassifier', X_train, y_train)


# In[14]:


#fitting the model selected from the best training
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.linear_model import LogisticRegression
from  xgboost import XGBClassifier

model= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


model=XGBClassifier(base_score=0.5, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, n_estimators=100,
       nthread=-1, objective='binary:logistic', seed=0, silent=True,
       subsample=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy of RandomForest classifier on test set: {:.2f}'.format(
    model.score(X_test, y_test)))

tune.write_result_class(X_test, y_test, y_pred,  model)


# In[15]:


# Do cross validation 
from sklearn import model_selection
print 'Perform 10 folder cross validation for the training data.'
kfold = model_selection.KFold(n_splits=10, random_state=7)
cv_results = model_selection.cross_val_score(model,X_train, y_train, cv=kfold, scoring='accuracy')
msg = "\n%s: \n\naccuracy=%f (std=%f)" % (model, cv_results.mean(), cv_results.std())
print(msg)


# In[16]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities
# store the predicted probabilities for class 1
from sklearn import metrics

y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print '\nSensitivity:', tpr[thresholds > threshold][-1]
    print 'Specificity:', 1 - fpr[thresholds > threshold][-1]

evaluate_threshold(0.5)



# 
# # Build a model to predict whether an activity is fraudulent or not. Explain how different assumptions about the cost of false positives vs false negatives would impact the model. 
# 
# Sensitivity (also called the true positive rate) measures the proportion of positives that are correctly identified as such.
# Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such. 
# Sensitivity therefore quantifies the avoiding of false negatives, and specificity does the same for false positives. 
# There is usually a trade-off between the two measures which can be adjusted by a threshold according to some business need.  High sensitivity may be needed for this case.
# 
# 
# # Your boss is a bit worried about using a model she doesn't understand for something as important as fraud detection. How would you explain her how the model is making the predictions? Not from a mathematical perspective (she couldn't care less about that), but from a user perspective. 
# 
# For classification problem, the accuracy and AUC are normally used to judge if the prediction is useful or not. In this case, since the trained data has much lower percentage of fraudulent observations than the non-fraudulent observations, the accuracy may not be very useful. For example, if all the non-fraudulent cases are correctly predicted, but not for the fraudulent, the accuracy value is still very high. Instead, the AUC value is more reliable. 
# 
# # What kinds of users are more likely to be classified as at risk? What are their characteristics? 
# 
# The data shows users from China have more fraudulent cases than users from other countries.
# 
# It is also interesting to note that the time difference between the purchase_time and signup_time  is significantly shorter for the  users who made fraud!
# 
# # From a product perspective, how would you use it? That is, what kind of different user experiences would you build based on the model output?!
# 
# The trained model is kept for late use. Once you have a new data set, you can input it to the model (y_pred = model.predict(X_test)) that was built using the already known observations.  The y_pred will give you 1 (fraudulent) or 0 (non-fraudulent).
# 
