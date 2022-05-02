#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np


# In[3]:


fig, ax=plt.subplots()
ax.plot([1,2,3,4], [1, 4, 2, 3])


# In[5]:


plt.plot([1, 2, 3, 4], [1, 4, 2, 3])


# In[6]:


fig=plt.figure()
fig, ax=plt.subplots()
fig, axs=plt.subplots(2,2)


# In[4]:


x=np.linspace(0,2,100)
fig, ax=plt.subplots()
ax.plot(x, x, label='linear')
ax.plot(x, x**2, label='quadratic')
ax.plot(x, x**3, label='cubic')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title("Simple Plot")
ax.legend()


# In[6]:


x=np.linspace(0, 2, 100)
plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()


# In[7]:


def my_plotter(ax, data1, data2, param_dict):
    out=ax.plot(data1, data2, **param_dict)
    return out


# In[8]:


data1, data2, data3, data4=np.random.randn(4, 100)
fig, ax=plt.subplots(1, 1)
my_plotter(ax, data1, data2, {'marker': 'x'})


# In[9]:


fig, (ax1, ax2)=plt.subplots(1, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})


# In[10]:


import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()


# In[11]:


plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


# In[12]:


plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()


# In[13]:


import numpy as np
t=np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# In[14]:


data={'a':np.arange(50), 
      'c':np.random.randint(0, 50, 50), 
      'd':np.random.randn(50)}
data['b']=data['a']+10*np.random.randn(50)
data['d']=np.abs(data['d'])*100
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()


# In[15]:


names=['group_a', 'group_b', 'group_c']
values=[1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorial Plotting')
plt.show()


# In[16]:


plt.plot(x, y, linewidth=2.0)


# In[17]:


lines=plt.plot([1, 2, 3])


# In[18]:


plt.setp(lines)


# In[19]:


def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)
t1=np.arange(0.0, 5.0, 0.1)
t2=np.arange(0.0, 5.0, 0.02)
plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


# In[21]:


plt.figure(1)
plt.subplot(211)
plt.plot([1, 2, 3])
plt.subplot(212)
plt.plot([4, 5, 6])
plt.figure(2)
plt.plot([4, 5, 6])
plt.figure(1)
plt.subplot(211)
plt.title('Easy as 1, 2, 3')


# In[22]:


mu, sigma=100, 15
x=mu+sigma*np.random.randn(10000)
n, bins, patches=plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100, \ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# In[25]:


ax=plt.subplot()
t=np.arange(0.0, 5.0, 0.01)
s=np.cos(2*np.pi*t)
line, =plt.plot(t, s, lw=2)
plt.annotate('local max', xytext=(3, 1.5), xy=(2, 1),  
             arrowprops=dict(facecolor='black', shrink=0.05)
            )
plt.ylim(-2, 2)
plt.show()


# In[26]:


np.random.seed(19680801)
y=np.random.normal(loc=0.5, scale=0.4, size=1000)
y=y[(y>0)&(y<1)]
y.sort()
x=np.arange(len(y))
plt.figure()
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)
plt.subplot(223)
plt.plot(x, y-y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
data={'Barton LLC': 109438.50,
         'Frami, Hills and Schmidt': 103569.59,
         'Frisch, Russel and Anderson': 112214.71, 
         'Jerde-Hilpert': 112591.43,
         'Keeling LLc': 100934.30,
         'Koepp Ltd': 103660.54, 
         'Kulas Inc':137351.96,
         'Trantow-Barrows':123381.38,
         'White-Trantow': 135841.99,
         'Will LLC': 104437.60}
group_data=list(data.values())
group_names=list(data.keys())
group_mean=np.mean(group_data)


# In[29]:


fig, ax=plt.subplots()
ax.barh(group_names, group_data)


# In[30]:


print(plt.style.available)


# In[31]:


plt.style.use('fivethirtyeight')


# In[39]:


plt.rcParams.update({'figure.autolayout': True})
fig, ax=plt.subplots(figsize=(8, 4))
ax.barh(group_names, group_data)
labels=ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company', 
      title='Company Revenue')


# In[40]:


def currency(x, pos):
    if x >=1e6:
        s='${:1.1f}M'.format(x*1e-6)
    else:
        s='${:1.0f}K'.format(x*1e-3)
    return s


# In[41]:


fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')

ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company',
       title='Company Revenue')
ax.xaxis.set_major_formatter(currency)


# In[44]:


fig, ax=plt.subplots(figsize=(8, 8))
ax.barh(group_names, group_data)
labels=ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.axvline(group_mean, ls='--', color='r')
for group in [3, 5, 8]:
    ax.text(145000, group, "New Company", fontsize=10,
           verticalalignment="center")
ax.title.set(y=1.05)
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company',
      title='Company Revenue')
ax.xaxis.set_major_formatter(currency)
ax.set_xticks([0, 25e3, 75e3, 100e3, 125e3])
fig.subplots_adjust(right=.1)
plt.show()


# In[45]:


print(fig.canvas.get_supported_filetypes())


# In[46]:


fig.savefig('sales.png', transparent=False, dpi=80, bbox_inches='tight')


# In[48]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(19680801)
data=np.random.randn(2, 100)
fig, axs=plt.subplots(2, 2, figsize=(5,5))
axs[0,0].hist(data[0])
axs[1,0].scatter(data[0], data[1])
axs[0,1].plot(data[0], data[1])
axs[1,1].hist2d(data[0], data[1])
plt.show()


# In[49]:


from IPython.core.display import HTML


# In[50]:


HTML('''
<h1>Hello DOM!</h1>
''')


# In[51]:


from IPython.core.display import display, HTML
from string import Template
import pandas as pd
import json, random


# In[52]:


HTML('<script src="https://d3js.org/d3.v4.min.js"></script>')


# In[ ]:




