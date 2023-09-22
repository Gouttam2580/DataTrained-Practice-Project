#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn



# In[15]:


data = pd.read_excel(r'C:\Users\dasgu\Desktop\Stock_Price.xlsx')


# In[16]:


data.head()


# In[17]:


data.info()


# In[18]:


data['Date'] = pd.to_datetime(data['Date'])


# In[19]:


print(f'Dataframe contains stock prices between {data.Date.min()}{data.Date.max()}')
print(f'Total days = {(data.Date.max() - data.Date.min()).days}days')


# In[20]:


data.describe()


# In[21]:


data[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# In[22]:


import plotly.graph_objs as go


# In[23]:


import plotly.graph_objs as go

layout = go.Layout(
    title='Stock Price of Data',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

data_data = [{'x': data['Date'], 'y': data['Close']}]
plot = go.Figure(data=data_data, layout=layout)



# In[11]:


plot.show()


# In[24]:


from sklearn.model_selection import train_test_split
#preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[25]:


#split the data into train and test sets
X = np.array(data.index).reshape(-1,1)
Y = data['Close']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)


# In[26]:


#feature scaling
scaler = StandardScaler().fit(X_train)


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[29]:


import plotly.graph_objs as go

trace0 = go.Scatter(
    x=X_train.T[0],
    y=Y_train,
    mode='markers',
    name='Actual'
)

trace1 = go.Scatter(
    x=X_train.T[0],
    y=lm.predict(X_train).T,
    mode='lines',
    name='Predicted'
)

data_data = [trace0, trace1]

layout = go.Layout(
    xaxis=dict(title='Day'),  # Specify the x-axis title
)

plot2 = go.Figure(data=data_data, layout=layout)


# In[ ]:


plot2.show()


# In[38]:


#calculate score for model evaluation
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train,lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train,lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''


# In[39]:


print(scores)


# In[ ]:




