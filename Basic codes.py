#!/usr/bin/env python
# coding: utf-8

# In[4]:



#Import the data and relevant libraries
import pandas as pd


# In[8]:


#sort data based on decreasing beer_servings and assign it to a variable called sort_beer
pd.set_option("display.max_columns",500)
pd.set_option('display.width',1000)
df_excel=pd.read_excel("alcohol-2.xlsx")


# In[12]:


sort_beer=df_excel.sort_values("beer_servings")
df_excel.sort_values("beer_servings",ascending=False)


# In[25]:


#which country drinks the highest spirit_servings
df_excel2=df_excel.loc[df_excel['spirit_servings'].idxmax()]
print(df_excel2.country)


# In[29]:


#get all rows with beer servings greater than 100
df_excel3=df_excel.loc[df_excel['beer_servings']>100]
print(df_excel3)


# In[30]:


#get all rows with beer servings greater than 100 in Asia (AS)
df_excel4=df_excel.loc[(df_excel['beer_servings']>100) & (df_excel['continent']=='AS')] 
print(df_excel4)


# In[31]:


#get all the rows for continent "A"
df_excel5=df_excel.loc[df_excel['continent']=='A']
print(df_excel5)


# In[32]:


#get the mean alcohol consumption per continent for every column (hint: use groupby)
df_excel6=df_excel.groupby('continent')


# In[40]:


df_excel6.first()
df_excel6.mean()


# In[41]:


#get the median alcohol consumption per continent for every column
df_excel6.median()


# In[49]:


#Create a new column called total_servings which is the sum of beer_servings, spirit_servings, wine_servings
df_excel['total_servings'] =df_excel['beer_servings']+df_excel['spirit_servings']+df_excel['wine_servings']
df_excel.head()


# In[65]:


#Sort the data based on total_servings and state which country drinks most and which drinks least
df_excel.sort_values("total_servings",ascending=False)


# In[63]:


df_excel7=df_excel.loc[df_excel['total_servings'].idxmax()]
print(df_excel7.country)


# In[105]:


#df_excel.query(df_excel.loc[df_excel['total_servings'].idxmax()])['country']
df_excel8=df_excel.loc[df_excel['total_servings'].idxmin()]
print(df_excel8.country)


# In[106]:


#Read column beer_servings
df_excel['beer_servings']


# In[110]:


#Read columns beer_servings and wine_servings
df_excel[['beer_servings','wine_servings']]


# In[124]:


#for countries that drink more than 200 servings of beer, change their (country names) names to "beer nation"
df_excel9=df_excel.loc[df_excel['beer_servings']>200,'country'] = 'Beer Nation'

df_excel.head()


# In[128]:


#save the data frame as an Excel file with name updated_drinks_excel
df_excel.to_excel('updated_drinks_excel.xlsx', index=False,header=True)


#save the data frame as a csv file with name updated_drinks_csv
df_excel.to_csv('updated_drinks_csv.csv', sep='\t', index=False,header=True)


# In[149]:


#Write a program to print the cube of numbers from 2 to 100 (including both 2 and 100)
listcube = []
for x in range(2,101):
    listcube.append(x*x*x)
print(listcube)


# In[170]:


#Write a program to print the cube of even numbers from 2 to 100 (including both 2 and 100)
for x in range(2,101):
    if x% 2 == 0:
        print(x*x*x, end = " ")
           
              


# In[163]:


#Give 5 examples of reserved words in python
# print, def, for, is, and


# In[ ]:


#give 4 examples of bad variable names and state why they are invalid
# 4nums is a bad Python variable name since started with numeric characters
# #num is a bad Python variable name since started with special character
# char.12 is a bad Python variable name since it has special character
# var1 = 10 Naming var1 will not show error sbut that does not make any sense and such type of variable name is a bad variable name

