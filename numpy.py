#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


numbers=np.array([2, 3, 5, 7, 11])


# In[3]:


type(numbers)


# In[4]:


numbers


# In[5]:


np.array([[1, 2, 3], [4, 5, 6]])


# In[6]:


import numpy as np
np.array([x for x in range(2, 21, 2)])


# In[8]:


np.array([[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]])


# In[9]:


import numpy as np
integers= np.array([[1, 2, 3], [4, 5, 6]])
integers


# In[10]:


floats=np.array([0.0, 0.1, 0.2, 0.3, 0.4])
floats


# In[11]:


integers.dtype


# In[12]:


floats.dtype


# In[13]:


integers.ndim


# In[14]:


floats.ndim


# In[15]:


integers.shape


# In[16]:


floats.shape


# In[17]:


integers.size


# In[18]:


integers.itemsize


# In[19]:


floats.size


# In[20]:


floats.itemsize


# In[21]:


for row in integers:
    for column in row:
        print(column, end=' ')
    print()


# In[22]:


for i in integers.flat:
    print(i, end=' ')


# In[24]:


a=np.array([[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]])
a.ndim


# In[25]:


a.shape


# In[26]:


np.zeros(5)


# In[28]:


np.ones((2,4), dtype=int)


# In[29]:


np.full((3, 5), 13)


# In[30]:


np.arange(5)


# In[31]:


np.arange(5, 10)


# In[32]:


np.arange(10, 1, -2)


# In[33]:


np.linspace(0.0, 1.0, num=5)


# In[34]:


np.arange(1, 21).reshape(4,5)


# In[35]:


np.arange(1, 100001).reshape(4, 25000)


# In[36]:


np.arange(1, 100001).reshape(100, 1000)


# In[37]:


np.arange(2, 41, 2).reshape(4, 5)


# In[38]:


import random


# In[39]:


get_ipython().run_line_magic('timeit', 'rolls_list=    [random.randrange(1,7) for i in range(0, 6_000_000)]')


# In[40]:


get_ipython().run_line_magic('timeit', 'rolls_array=np.random.randint(1, 7, 6_000_000)')


# In[41]:


get_ipython().run_line_magic('timeit', 'rolls_array=np.random.randint(1, 7, 60_000_000)')


# In[42]:


get_ipython().run_line_magic('timeit', 'rolls_array=np.random.randint(1, 7, 600_000_000)')


# In[43]:


get_ipython().run_line_magic('timeit', '-n3 -r2 rolls_array=np.random.randint(1, 7, 6_000_000)')


# In[44]:


get_ipython().run_line_magic('timeit', 'sum([x for x in range(10_000_000)])')


# In[45]:


get_ipython().run_line_magic('timeit', 'np.arange(10_000_000).sum()')


# In[46]:


numbers=np.arange(1, 6)
numbers


# In[47]:


numbers * 2


# In[48]:


numbers ** 3


# In[49]:


numbers


# In[50]:


numbers +=10


# In[51]:


numbers


# In[52]:


numbers2=np.linspace(1.1, 5.5, 5)


# In[53]:


numbers2


# In[54]:


numbers * numbers2


# In[55]:


numbers


# In[56]:


numbers >=13


# In[57]:


numbers2


# In[58]:


numbers2<numbers


# In[59]:


numbers==numbers2


# In[60]:


numbers==numbers


# In[61]:


np.arange(1, 6)**2


# In[64]:


grades=np.array([[87, 96, 70], [100, 87, 90],
                [94, 77, 90], [100, 81, 82]])
grades


# In[66]:


grades.sum()


# In[67]:


grades.min()


# In[68]:


grades.max()


# In[69]:


grades.mean()


# In[70]:


grades.std()


# In[71]:


grades.var()


# In[72]:


grades.mean(axis=0)


# In[74]:


grades.mean(axis=1)


# In[75]:


grades2=np.random.randint(60, 101, 12).reshape(3,4)


# In[76]:


grades


# In[77]:


grades2


# In[78]:


grades2.mean()


# In[79]:


grades2.mean(axis=0)


# In[80]:


grades2.mean(axis=1)


# In[81]:


numbers=np.array([1, 4, 9, 16, 25, 36])


# In[82]:


numbers


# In[83]:


np.sqrt(numbers)


# In[84]:


numbers2=np.arange(1, 7)*10


# In[85]:


numbers2


# In[86]:


np.add(numbers, numbers2)


# In[87]:


np.multiply(numbers2, 5)


# In[88]:


numbers3=numbers2.reshape(2,3)


# In[89]:


numbers3


# In[90]:


numbers4=np.array([2, 4, 6])


# In[91]:


np.multiply(numbers3, numbers4)


# In[92]:


numbers=np.arange(1, 6)


# In[93]:


np.power(numbers, 3)


# In[94]:


grades=np.array([[87, 96, 70], [100, 87, 90],
                [94, 77, 90], [100, 81, 82]])


# In[95]:


grades


# In[96]:


grades[0, 1]


# In[97]:


grades[1]


# In[98]:


grades[0:2]


# In[99]:


grades[[1,3]]


# In[100]:


grades[:,0]


# In[101]:


grades[:, 1:3]


# In[102]:


grades[:, [0, 2]]


# In[103]:


a=np.arange(1, 16).reshape(3, 5)


# In[104]:


a


# In[105]:


a[1]


# In[107]:


a[[0, 2]]


# In[108]:


a[:, 1:4]


# In[109]:


numbers=np.arange(1,6)
numbers


# In[110]:


numbers2=numbers.view()


# In[111]:


numbers2


# In[112]:


id(numbers)


# In[113]:


id(numbers2)


# In[114]:


numbers[1]*=10


# In[115]:


numbers2


# In[116]:


numbers


# In[117]:


numbers2[1]/=10


# In[118]:


numbers


# In[119]:


numbers2


# In[120]:


numbers2=numbers[0:3]


# In[121]:


numbers2


# In[122]:


id(numbers)


# In[123]:


id(numbers2)


# In[124]:


numbers2[3]


# In[125]:


numbers[1]*=20


# In[126]:


numbers


# In[127]:


numbers2


# In[128]:


numbers=np.arange(1, 6)
numbers


# In[129]:


numbers2=numbers.copy()


# In[130]:


numbers2


# In[131]:


numbers[1]*=10


# In[132]:


numbers


# In[133]:


numbers2


# In[134]:


grades=np.array([[87, 96,70], [100, 87, 90]])
grades


# In[135]:


grades.reshape(1, 6)


# In[136]:


grades


# In[137]:


grades.resize(1,6)


# In[138]:


grades


# In[139]:


grades=np.array([[87, 96, 70], [100, 87, 90]])


# In[140]:


grades


# In[141]:


flattened=grades.flatten()


# In[142]:


grades


# In[143]:


flattened[0]=100


# In[144]:


flattened


# In[145]:


grades


# In[146]:


raveled=grades.ravel()


# In[147]:


raveled


# In[148]:


grades


# In[149]:


raveled[0]=100


# In[150]:


raveled


# In[151]:


grades


# In[152]:


grades.T


# In[153]:


grades


# In[154]:


grades2=np.array([[94, 77, 90], [100, 81, 82]])


# In[155]:


np.stack((grades, grades2))


# In[156]:


a=np.arange(1, 7).reshape(2, 3)


# In[157]:


a=np.hstack((a, a))


# In[158]:


a=np.vstack((a, a,))


# In[159]:


a


# In[160]:


import pandas as pd


# In[161]:


grades=pd.Series([87, 100, 94])


# In[162]:


grades


# In[163]:


pd.Series(98.6, range(3))


# In[164]:


grades[0]


# In[165]:


grades.count()


# In[166]:


grades.mean()


# In[167]:


grades.min()


# In[168]:


grades.max()


# In[169]:


grades.std()


# In[170]:


grades.describe()


# In[171]:


grades=pd.Series([87, 100, 94], index=['Wally', 'Eva', 'Sally'])


# In[172]:


grades


# In[173]:


grades=pd.Series({'Wally':87, 'Eva':100, 'Sam': 94})


# In[174]:


grades


# In[175]:


grades['Eva']


# In[176]:


grades.Wally


# In[177]:


grades.dtype


# In[178]:


grades.values


# In[179]:


hardware=pd.Series(['Hammer', 'Saw', 'Wrench'])


# In[180]:


hardware


# In[181]:


hardware.str.contains('a')


# In[182]:


hardware.str.upper()


# In[183]:


temps=np.random.randint(60, 101, 6)
temperatures=pd.Series(temps)
temperatures


# In[184]:


temperatures.min()


# In[185]:


temperatures.max()


# In[186]:


temperatures.mean()


# In[187]:


temperatures.describe()


# In[188]:


grades_dict={'Wally':[87, 96, 70], 'Eva':[100, 87, 90], 'Sam':[94, 77, 90], 'Kate':[100, 81, 82], 'Bob':[83, 65, 85]}


# In[189]:


grades=pd.DataFrame(grades_dict)


# In[190]:


grades


# In[191]:


grades.index=['Test1', 'Test2', 'Test3']


# In[192]:


grades


# In[193]:


grades['Eva']


# In[194]:


grades.Sam


# In[195]:


grades.loc['Test1']


# In[196]:


grades.iloc[1]


# In[197]:


grades.loc['Test1':'Test3']


# In[198]:


grades.iloc[0:2]


# In[199]:


grades.loc[['Test1', 'Test3']]


# In[200]:


grades.iloc[[0, 2]]


# In[203]:


grades.loc['Test1':'Test2', [('Eva'), ('Kate')]]


# In[204]:


grades.iloc[[0,2], 0:3]


# In[205]:


grades[grades>=90]


# In[206]:


grades[(grades>=80) & (grades<90)]


# In[207]:


grades.at['Test2', 'Eva']


# In[208]:


grades.iat[2,0]


# In[209]:


grades.at['Test2', 'Eva']=100


# In[210]:


grades.at['Test2', 'Eva']


# In[211]:


grades.iat[1,2]=87


# In[212]:


grades.iat[1,2]


# In[213]:


grades.describe()


# In[214]:


pd.set_option('precision', 2)


# In[215]:


grades.describe()


# In[216]:


grades.mean()


# In[217]:


grades.T


# In[218]:


grades.T.describe()


# In[219]:


grades.T.mean()


# In[220]:


grades.sort_index(ascending=False)


# In[221]:


grades.sort_index(axis=1)


# In[222]:


grades.sort_values(by='Test1', axis=1, ascending=False)


# In[223]:


grades.T.sort_values(by='Test1', ascending=False)


# In[224]:


grades.loc['Test1'].sort_values(ascending=False)


# In[225]:


temps={'Mon':[68, 89], 'Tue':[71, 93], 'Wed':[66, 82], 'Thu':[75, 97], 'Fri':[62,79]}
temperatures=pd.DataFrame(temps, index=['Low', 'High'])


# In[226]:


temperatures


# In[228]:


temperatures.loc[:, 'Mon':'Wed']


# In[229]:


temperatures.loc['Low']


# In[230]:


pd.set_option('precision', 2)


# In[231]:


temperatures.mean()


# In[232]:


temperatures.mean(axis=1)


# In[ ]:




