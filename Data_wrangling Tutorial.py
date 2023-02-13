#!/usr/bin/env python
# coding: utf-8

# In[318]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# In[320]:


#Check the number of maximum returned rows that can be displayed at once
pd.options.display.max_rows


# In[321]:


df = pd.read_csv("C:\\Users\\HP\\Downloads\\automoblile.csv")


# In[322]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[323]:


#df = df.drop(df.index[[0]], axis=0)
df.columns=headers


# In[324]:


#Increase the number of maximum returned rows that can be displayed at once
pd.options.display.max_rows = 500
df


# In[329]:


df.replace('?', np.nan, inplace = True)
df


# In[330]:


df.dtypes


# In[331]:


missing_data = df.isnull()


# In[332]:


missing_data.head()


# In[333]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# In[334]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# In[335]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# In[336]:


df["normalized-losses"]


# In[337]:


avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# In[338]:


df['bore'].replace(np.nan, avg_bore, inplace=True)


# In[339]:


df


# In[241]:


avg_stroke = df['stroke'].astype('float').mean(axis=0)
print('Average of stroke:', avg_stroke)


# In[242]:


df['stroke'].replace(np.nan, avg_stroke, inplace=True)


# In[224]:


df.head(20)


# In[340]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print('mean of horsepower:', avg_horsepower)


# In[341]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# In[342]:


df


# In[343]:


avg_peak_rpm=df['peak-rpm'].astype('float').mean(axis=0)


# In[344]:


print('Mean of peak-rpm:', avg_peak_rpm)


# In[345]:


df['peak-rpm'].replace(np.nan, avg_peak_rpm, inplace=True)


# In[346]:


df


# In[347]:


#To see which values are present in a particular column, we can use the ".value_counts()" method:

df['num-of-doors'].value_counts()


# In[348]:


#We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate the most common type:

df['num-of-doors'].value_counts().idxmax()


# In[250]:


df['num-of-doors'].replace(np.nan, 'four', inplace=True)


# In[251]:


df.dropna(subset=['price'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


# In[99]:


df


# In[255]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["price"] = df["price"].astype("float")
df["peak-rpm"] = df["peak-rpm"].astype("float")


# In[257]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()


# In[258]:


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-L/100km"] = 235/df["highway-mpg"]


# In[259]:


# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={"highway-L/100km":'highway-mpg'}, inplace=True)


# In[260]:


df


# In[273]:


#normalization of 'lenght', 'width', 'height' (simple normalization scalling method)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
df[['length','width','height']]


# In[274]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[289]:



import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[290]:


#We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[291]:


group_names = ['Low', 'Medium', 'High']


# In[292]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[293]:


df["horsepower-binned"].value_counts()


# In[294]:


import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[295]:


#Normally, a histogram is used to visualize the distribution of bins we created above
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[298]:


#We use indicator variables so we can use categorical variables for regression analysis
#We see the column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, only numbers.
#To use this attribute in regression analysis, we convert "fuel-type" to indicator variables
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[299]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In[301]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[304]:


df.head()


# In[306]:


# get indicator variables of aspiration and assign it to data frame
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.head(10)


# In[307]:


dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)


# In[309]:


df.head()


# In[310]:


# merge the new dataframe to the original dataframe 
df = pd.concat([df, dummy_variable_2], axis = 1)


# In[312]:


df.head()


# In[313]:


# drop original column "aspiration" from 'df'
df.drop('aspiration', axis=1, inplace=True)


# In[314]:


df.head()


# In[317]:


df.to_csv('new_clean_data.csv')


# In[ ]:




