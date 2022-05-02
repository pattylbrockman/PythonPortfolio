#!/usr/bin/env python
# coding: utf-8

# In[6]:


conda install -c anaconda psycopg2


# In[7]:


import psycopg2
psycopg2.connect(host="localhost", user="postgres", password="redacted", dbname="postgres", port=5433)


# In[8]:


from sqlalchemy import create_engine
import pandas as pd


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


cnxn_string=("postgresql+psycopg2://{username}:{pswd}"
             "@{host}:{port}/{database}")
print(cnxn_string)


# In[11]:


engine=create_engine(cnxn_string.format(
    username="postgres",
    pswd="Pattyls9!",
    host="localhost",
    port=5433,
    database="postgres"))


# In[13]:


engine.execute("select c.name, count(l.language) from countrylanguage l inner join country c on c.code=l.countrycode where isofficial='T' group by c.name having count(l.language)>2 order by count desc").fetchall()


# In[15]:


top_off_languages=pd.read_sql_table('countrylanguage', engine)


# In[36]:


query="""
Select c.name, count(l.language)
from countrylanguage l
inner join country c on c.code=l.countrycode
where isofficial='T'
group by c.name
having count(l.language)>2
order by count desc
"""


# In[37]:


top_off_languages=pd.read_sql_query(query, engine)


# In[38]:


top_off_languages


# In[40]:


ax=top_off_languages.plot.bar('name', y='count',title='Top Number of Official Languages')


# In[ ]:




