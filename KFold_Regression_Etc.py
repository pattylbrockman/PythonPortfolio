#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[116]:


nyc=pd.read_csv('desktop/ave_hi_nyc_jan_1895-2018.csv')


# In[117]:


nyc.head()


# In[118]:


nyc.tail()


# In[119]:


nyc.columns=['Date', 'Temperature', 'Anomaly']


# In[120]:


nyc.head(3)


# In[121]:


nyc.Date.dtype


# In[122]:


nyc.Date=nyc.Date.floordiv(100)


# In[123]:


nyc.head(3)


# In[124]:


pd.set_option('precision', 2)


# In[125]:


nyc.Temperature.describe()


# In[126]:


from scipy import stats


# In[127]:


linear_regression=stats.linregress(x=nyc.Date, y=nyc.Temperature)


# In[128]:


linear_regression.slope


# In[129]:


linear_regression.intercept


# In[130]:


linear_regression.slope* 2021 + linear_regression.intercept


# In[131]:


linear_regression.slope* 1890 + linear_regression.intercept


# In[132]:


import seaborn as sns


# In[133]:


sns.set_style('whitegrid')


# In[134]:


axes=sns.regplot(x=nyc.Date, y=nyc.Temperature)


# In[23]:


from sklearn.datasets import load_digits


# In[24]:


digits=load_digits()


# In[25]:


print(digits.DESCR)


# In[26]:


digits.target[::100]


# In[27]:


digits.data.shape


# In[28]:


digits.target.shape


# In[29]:


digits.images[13]


# In[30]:


digits.data[13]


# In[31]:


digits.images[22]


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


figure, axes=plt.subplots(nrows=4, ncols=6, figsize=(6,4))


# In[34]:


for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target=item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split( digits.data, digits.target, random_state=11)


# In[37]:


X_train.shape


# In[38]:


X_test.shape


# In[39]:


from sklearn.neighbors import KNeighborsClassifier


# In[40]:


knn=KNeighborsClassifier()


# In[41]:


knn.fit(X=X_train, y=y_train)


# In[42]:


predicted=knn.predict(X=X_test)


# In[43]:


expected=y_test


# In[46]:


predicted[:20]


# In[47]:


expected[:20]


# In[48]:


wrong=[(p, e) for (p, e) in zip(predicted, expected) if p != e]


# In[49]:


wrong


# In[50]:


print(f'{knn.score(X_test, y_test):.2%}')


# In[51]:


from sklearn.metrics import confusion_matrix


# In[53]:


confusion=confusion_matrix(y_true=expected, y_pred=predicted)


# In[54]:


confusion


# In[55]:


from sklearn.metrics import classification_report


# In[56]:


names=[str(digit) for digit in digits.target_names]


# In[57]:


print(classification_report(expected, predicted, target_names=names))


# In[58]:


import pandas as pd


# In[59]:


confusion_df= pd.DataFrame(confusion, index=range(10), columns=range(10))


# In[60]:


import seaborn as sns


# In[61]:


axes=sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')


# In[62]:


from sklearn.model_selection import KFold


# In[63]:


kfold=KFold(n_splits=10, random_state=11, shuffle=True)


# In[64]:


from sklearn.model_selection import cross_val_score


# In[65]:


scores=cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)


# In[66]:


scores


# In[67]:


print(f'Mean accuracy: {scores.mean():.2%}')


# In[68]:


print(f'Accuracy standard deviation: {scores.std():.2%}')


# In[69]:


from sklearn.svm import SVC


# In[70]:


from sklearn.naive_bayes import GaussianNB


# In[73]:


estimators={'KNeighborsClassifier':knn, 'SVC': SVC(gamma='scale'), 'GaussianNB': GaussianNB()}


# In[74]:


for estimator_name, estimator_object in estimators.items():
    kfold=KFold(n_splits=10, random_state=11, shuffle=True)
    scores=cross_val_score(estimator=estimator_object,
                          X=digits.data, y=digits.target, cv=kfold)
    print(f'{estimator_name:>20}: '+
         f'mean accuracy={scores.mean():.2%}; ' +
         f'standard deviation={scores.std():.2%}')


# In[75]:


for k in range(1, 20, 2):
    kfold=KFold(n_splits=10, random_state=11, shuffle=True)
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(estimator=knn, 
                          X=digits.data, y=digits.target, cv=kfold)
    print(f'k={k:<2}; mean accuracy={scores.mean():.2%}; ' +
         f'standard deviation={scores.std():.2%}')


# In[76]:


import pandas as pd


# In[78]:


nyc=pd.read_csv('desktop/ave_hi_nyc_jan_1895-2018.csv')


# In[79]:


nyc.columns=['Date', 'Temperature', 'Anomaly']


# In[80]:


nyc.Date=nyc.Date.floordiv(100)


# In[81]:


nyc.head(3)


# In[82]:


from sklearn.model_selection import train_test_split


# In[89]:


X_train, X_test, y_train, y_test=train_test_split(nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11)


# In[90]:


X_train.shape


# In[85]:


X_test.shape


# In[92]:


from sklearn.linear_model import LinearRegression


# In[93]:


linear_regression=LinearRegression()


# In[94]:


linear_regression.fit(X=X_train, y=y_train)


# In[95]:


linear_regression.coef_


# In[96]:


linear_regression.intercept_


# In[98]:


predicted=linear_regression.predict(X_test)


# In[99]:


expected=y_test


# In[100]:


for p, e in zip(predicted[::5], expected[::5]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}')


# In[103]:


predict=(lambda x:linear_regression.coef_ * x +
        linear_regression.intercept_)


# In[104]:


predict(2019)


# In[105]:


predict(1890)


# In[106]:


import seaborn as sns


# In[108]:


axes=sns.scatterplot(data=nyc, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)


# In[109]:


axes.set_ylim(10, 70)


# In[110]:


import numpy as np


# In[111]:


x=np.array([min(nyc.Date.values), max(nyc.Date.values)])


# In[112]:


y=predict(x)


# In[113]:


import matplotlib.pyplot as plt


# In[114]:


line=plt.plot(x, y)


# In[ ]:




