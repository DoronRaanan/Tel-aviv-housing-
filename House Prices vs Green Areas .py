#!/usr/bin/env python
# coding: utf-8

# ### During this exercise I will try to examine whether the percentage of areas in the neighborhood allocated to green areas affects the price of properties in general and the rate of increase in prices in particular.

# #### When plotting please change the path to the path  where you want to save your files 

# #### Please upload the provided data files to the same folder as the notebook (The files are inside folders- upload all files inside each folder)

# ## Packages and Settings 

# In[1]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import geopandas as gpd
from os import listdir
import pyproj
from shapely.geometry import Point
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import json


# ## Importing the data

# ###### Now I will load the data which is currently csv or shp into geo/dataframes

# ### importing the sales data

# In[2]:


sales = pd.read_csv('AssetsDealsTLV20122017.csv', # name of the csv
                    index_col=0) # Column to use as the row labels of the DataFrame


# ### Importing city data 

# ###### note to self - send this data with the project

# In[3]:


### This shp files are extracted from Tel Aviv`s GIS website
##Neighbourhoods - This data includes Tel Aviv`s Neighbourhoods boundaries, Geodata as polygons 
NBHD = gpd.read_file('Neighbourhoods.shp')
#Green - This file includes data on Green areas and public parks in Tel Aviv, Geodata as polygons  
Green = gpd.read_file(r'Green/Green Areas and Public Parks.shp')


# ## Exploring the data

# ### for each dataframe Showing the data`s head (first 5 rows) to see a preview

# In[4]:


sales.head()


# In[5]:


Green.head()


# In[6]:


NBHD.head()


# ## Preliminary information processing

# ### Working on Green

# #### Fixing invalid data

# ##### get a list of all invalid polygons 

# In[7]:


Green[Green.is_valid== False] # get only those polygons where is_valid function yields false 


# In[8]:


Green['valid'] = Green.is_valid # add a column  stating whether the polygon is valid 
Green # now there is a new column added stating whether the polygon is valid


# ##### Trying to understand why polygons are invalid - looking at park hayarkon

# In[9]:


# creating a 2 on 2 figure on which I will draw valid and invalid polygons 
# the axes on the figure will share x and y and will be on a constrained layout
# figure size will be 15,8
fig, ax = plt.subplots(2,2, figsize = (15,8), sharex=True, sharey=True, constrained_layout=True)

#setting main title and size for the figure
fig.suptitle('Graph 1: Plotting Park Hayarkon To Find Invalid Issue Source', fontsize=35) 

# drawing all valid polygons in park hayarkon, each in a diffrent color
#in unordered color scheme to see the difference between each other
Green[(Green.shemgan== 'פארק הירקון') & (Green.is_valid== True)].plot(column='UniqueId', ax= ax[0, 0])
ax[0, 0].set_title('Valid polygons in park Hayarkon', size = 25) # setting a title and size 

# drawing all invalid polygons in park hayarkon, in unordered color scheme to see the diffrence between each other
Green[(Green.shemgan== 'פארק הירקון') & (Green.is_valid== False)].plot(column='UniqueId', ax= ax[0,1])
ax[0,1].set_title('Invalid polygons in park Hayarkon', size = 25)# setting a title and size 

# drawing all  polygons in park hayarkon, in unordered color scheme to see the diffrence between each other
Green[(Green.shemgan== 'פארק הירקון')].plot(column='UniqueId', ax= ax[1,1])
ax[1,1].set_title('All polygons in park Hayarkon', size = 25)# setting a title and size 

# drawing all polygons in park hayarkon, colored by vaility
Green[(Green.shemgan== 'פארק הירקון')].plot(column='valid', ax= ax[1, 0])
ax[1,0].set_title('All polygons according to validity', size = 25)# setting a title and size 


# save with dpi 150 
fig.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Hayarkon_invalid_issue_source.png', dpi=150)


# ###### here we can see that at least some of the polygons are diffrent from the valid polygons by having holes in them

# ##### I will try to fix the polygons with buffer(0) method

# In[10]:


Green.geometry = Green.geometry.buffer(0)


# ##### Checking the validity of the data now

# In[11]:


Green[Green.is_valid== False] # only get those polygons where is_valid function yields false


# ##### Green data is valid now 

# In[12]:


# re-saving green with 'valid' droped because it's irrelevant now that evreything is valid
Green = Green.drop(columns = 'valid')


# ####  projections 

# In[13]:


# chcking if projections are the same on both layers so that geographical processing between them will be correct 
print(NBHD.crs) # projection on NBHD 
print(Green.crs) # projection on Green
print(Green.crs == NBHD.crs) # are they the same ? 


# ### Inital work on data - Sales 

# #### price per meter 

# In[14]:


# calculating the price per area for each assest
sales['pricepermeter'] = (sales['Deals_DealAmount']) /  sales['Assets_AssetArea'] 
# doing this in order to work on normalized data 


# #### Geodataframe

# In[15]:


type(sales) # data is a dataframe and not geodataframe


# ##### Converting into Geodataframe

# In[16]:


#list of all points as a point shape
geometry = [Point(xy) for xy in zip(sales.Assets_X_RECOMMENDED, sales.Assets_Y_RECOMMENDED)]
# the correct projection as checked earlier (same as the other dataset)
crs = ('epsg:2039') 
# creating a geodataframe with the new points and the correct projection 
geo_sales = gpd.GeoDataFrame(sales, crs=crs, geometry=geometry) 


# ###### making sure that evreything went correctly

# In[17]:


geo_sales.head() # we have a new geometry column


# In[18]:


type(geo_sales) # this is a geodataframe


# In[19]:


geo_sales.crs # correct projection


# In[20]:


geo_sales[geo_sales.is_valid==False] # all data is valid 


# #### Adding a year to the data

# In[21]:


geo_sales['Deals_DealDate'] = pd.to_datetime(geo_sales['Deals_DealDate']) # converting date into datetime type 


# ##### getting only the year into a new column

# In[22]:


# new column on the dataframe called year will be equal to datetime series year attribute type has a year attribute
geo_sales['Year'] = geo_sales['Deals_DealDate'].apply(lambda x: x.year) # datetime type has a year attribute
#this will be usefull later


# #### Cleaning the geo_sales data

# ##### checking if all columns are relevent

# In[23]:


geo_sales.columns # a list of all the columns 


# In[24]:


# Trying to understand why is there a "Assets_SettlementNameEng" column and how often and when its being used 
Settl = sns.catplot(x="Assets_SettlementNameEng", y="Year", data=geo_sales) # plotting the data
#setting titles 
Settl.set(
    title ="Graph 2: Records From Each Settlement Each Year",
    xlabel= 'Assets Settlement Name',
    ylabel='Year' ) 
#save fig as png 
Settl.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\settel.png', dpi=150 ) 


# In[25]:


list_to_drop= ['Deals_DealDate', # only care about the year 
               'Assets_SettlmentID', # its all in Tel aviv
               'Assets_SettlementNameHeb', # all in Tel aviv
               'Assets_SettlementNameEng', # all in Tel aviv
               'Assets_X_RECOMMENDED', #already used in geometry
               'Assets_Y_RECOMMENDED', # already used in geometry
               'HUJIUnite' # irrelevent
                                 ]


# In[26]:


geo_sales = geo_sales.drop(columns=list_to_drop) # drop those irrelvent columns


# In[27]:


geo_sales.head() # see the new head 


# ## Green area by neighborhood

# #### for each green area check within which neighbourhoods is it and state it. If needed dived the green area into few neighbourhoods

# In[28]:


GreenNBHD = gpd.overlay(NBHD, Green, how='intersection')


# ###  Validating the overlay 

# #### make sure that overlay divded it by neighbourhood - plot

# In[29]:


fig, ax = plt.subplots(1,1, figsize = (15,20)) #  subplots = 1 on 1 matrix of axes, figsize 15*20 

ax.set_title('Graph 3: Green Area Divided by Neighbourhoods', size = 35) # set axes title and size
ax.set(facecolor = "white") # white background 
ax.grid(0.7, color= "gray", alpha= 0.4) # setting grid 
NBHD.boundary.plot(ax= ax,  edgecolor='black') # show the neighbourhoods boundray
GreenNBHD.plot(ax= ax,column = 'msshchuna', cmap = "prism") # plot each greenarea by the color of the neighbourhood its in
# This way we will see if those grean areas on that are split between neighbourhoods were split during overlay
# save
fig.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Green_area_neighbourhoods.png', dpi=150)


# #### here we can see that each new green area got the relvent neighbourhood and was divded if needed

# ### Area 

# ##### check if they contain the same sum of area 

# In[30]:


eacheck  = Green.area.sum() - GreenNBHD.area.sum() #sum of green minus sum of GreenNBHD


# In[31]:


eacheck # There are some missing areas 


# ### Unique Id check

# ##### check if all green unique Id that exists in Green also exist in GreenNBHD

# In[32]:


GreenIdscheck = Green.loc[~Green['UniqueId'].isin(GreenNBHD['UniqueId_2'])] 
# GreenIdscheck is a dataframs containing ids which are on Green but not in the new dataframe GreenNBHD


# In[33]:


GreenIdscheck


# #### We can see that there are some polygons which exist in Green and not in GreenNBHD and that some of them are empty

# In[34]:


GreenIdscheck.area.sum() # those that are not empty explain at least some of the missing area


# ##### Plotting the polygons in order to understand why they dont exist in the new dataframe 

# In[35]:


fig, ax = plt.subplots(1,1, figsize = (15,20)) #  subplots = 1 on 1 matrix of axes, figsize 15* 20 , call axis ax 

#Plot neighbourhoods boundary
NBHD.boundary.plot(ax= ax,  edgecolor='black')

# Plot those green areas that do not exist in our new dataframe GreenNBHD
GreenIdscheck.plot(ax=ax, 
            color = 'red')

ax.set_title('Graph 4: Green Area Outside of Neighborhoods', size = 35) # setting title and textsize
ax.set(facecolor = "white") # set background
ax.grid(0.7, color= "gray", alpha= 0.4) # setgrid

fig.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\FreenoutsideNBHD.png', dpi=150) # save 


# ###### Here we can see that the polgons that didnt overlay are just outside the city boundreys 

# ### Total Green Area By msshchuna(neighbourhoods) - grouping

# In[36]:


GreenNBHD['Garea'] = GreenNBHD.area # set a new column with the green area


# ###### getting this into a Geojson

# In[37]:


GreenNBHD.to_file("GreenNBHD.json", driver="GeoJSON") 


# In[38]:


summery = GreenNBHD.groupby('msshchuna').sum('Garea') # sum the Garea by neighbourhoods


# In[39]:


summery # each msshchuna got its relevent sum of green area value


# ### Merging

# #### merging stat data with NBHD in order to get the statistics and the geodata on the same table

# In[40]:


AreaStat= NBHD.merge(summery, on = 'msshchuna')


# ### Normlizing the greenarea sum

# In[41]:


AreaStat['Narea']= AreaStat.area # getting each neighbourhoods' area 


# #### normelize the green area with the normal area

# In[42]:


AreaStat['greenAreaInNBHD'] = (AreaStat.Garea / AreaStat.Narea) *100  # normelized green area by neighbourhood area


# ###### check if any is bigger than 100 because thats not possible

# In[43]:


AreaStat[AreaStat.greenAreaInNBHD > 100 ] 


# ### Ploting 

# In[44]:


# now I will plot each neighbourhood by its normalizd green value 

fig, ax = plt.subplots(1, figsize = (15,10)) #  subplots = 1 on 1 matrix of axes, figsize 10* 15, call axis ax 

#getting the neighbourhood name on the polygon
AreaStat.apply(lambda x: ax.annotate(text=x.shemshchun[11::-1], xy=x.geometry.centroid.coords[0], ha='center', fontsize=8, color= 'black'),axis=1)

#Plot AreaStat colored by greenAreaInNBHD 
AreaStat.plot(ax= ax, 
              column = 'greenAreaInNBHD', 
              cmap = 'Greens',  edgecolor='black', 
              legend = True, linewidth=2)

#setting a title 
ax.set_title('Graph 5: Total Green Area in Each Neighborhood \n Normalized to The Area of The Neighborhood', size = 24)
# creating a grid
plt.grid( color='grey',  alpha=0.3)
# fitting the page 
plt.tight_layout()

# save
fig.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\total_green_area_neighborhood_normalized.png', dpi=150)


# ## Cleaning the data 

# ### Before working on sales lets see if  all columns on AreaStat are needed

# In[45]:


tuple(AreaStat.columns) # create a tuple from the list of all the columns 


# ### Lest drop all those columns which are useless now

# In[46]:


list_to_drop= [
                                 'dateimport', # data is irrelvent
                                'ShapeArea_x', # alredy have shape area column
                                 'ShapeArea_y',  # alredy have shape area column
                                  'oidshetach', #not used 
                                  'msshefa', # not used 
                                  'oidshchuna_y',  # msshchuna is used
                                  'oidshchuna_x',
                                    'swnagish', # irrelvent
                                    'msarea' # area was calculated already 
                                 ]


# In[47]:


AreaStat = AreaStat.drop(columns=list_to_drop) # dropping those columns


# In[48]:


# looking at the data now 
colplot = sns.pairplot(data=AreaStat) # plotint the series as pairplot
colplot.fig.suptitle(
  "Graph 6: Green and Neighborhood Columns as Pairplot", # title text
    verticalalignment='top', y = 1.03, fontsize = 25) #Title location
colplot.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\AreaStatcol.png', dpi=150 )   # save


# ## Geo work on sales 

# ### checking on which neighborhood each sale happened 

# ##### Spatial Join geo_sales and  NBHD by what neighborhood contains what geo_sales

# In[49]:


geo_salesNBHD = gpd.sjoin(NBHD,geo_sales, how = 'inner', op = 'contains') # for each sale - what neighborhood contains it


# In[50]:


geo_salesNBHD


# ### mean price by neighborhoods

# In[51]:


# grouping the data by msshchuna to get mean price in each neighborhoods
meanPricePerNBHD = geo_salesNBHD.groupby('msshchuna').mean('pricepermeter')  
meanPricePerNBHD


# ### merging the meanPricePerNBHD with the AreaStat data
# #### I want total green area and mean priceper meter in each neighborhood on the same dataset

# In[52]:


NBHDpriceStat= AreaStat.merge(meanPricePerNBHD, on = 'msshchuna') #


# ### plotting - mean price per neighborhood

# In[53]:


fig, ax = plt.subplots(1, figsize = (15,15)) #  subplots = 1 on 1 matrix of axes, figsize 15* 15, call axis ax 

#Adding neighborhood name to the plot 
NBHDpriceStat.apply(lambda x: ax.annotate(text=x.shemshchun[11::-1], xy=x.geometry.centroid.coords[0], ha='center', fontsize=10, color= 'deeppink'),axis=1)

#Plot NBHDpriceStat
NBHDpriceStat.plot(ax= ax, 
              column = 'pricepermeter', 
              cmap = 'Blues',  edgecolor='black', 
              legend = True, linewidth=2)


#setting a title and textsize
ax.set_title('Graph 7: Mean Price per Meter by Neighborhood', size = 35)

# creating a grid
plt.grid( color='grey',  alpha=0.3)

# fitting the page 
plt.tight_layout()

#save
fig.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Mean_price_per_meter_neighborhood.png', dpi=150)


# ##  Prices vs green areas - getting to the question 

# ### first we will see if there is connection between prices and amount of green areas in the neighborhood

# In[54]:


# scatterplotting 'pricepermeter' vs 'greenAreaInNBHD' 
# seeing the mean price in each neighborhood vs the amount of green areas in it 
# plotting it with linear regression model fit.
ploti = sns.regplot(x="greenAreaInNBHD", y="pricepermeter", data=NBHDpriceStat) # plotting the data
ploti.set(
    title ="Graph 8: Mean Price vs Amount of Green Areas \n In Each Neighborhood",
    xlabel= 'Green area normalized to neighborhood`s area',
    ylabel='Mean Price per meter for asset') # set titles 
#save fig as png 
ploti.figure.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Mean_price_vs_Green_area.png', dpi=150 ) 


# In[55]:


# check correlation between those two series
print(NBHDpriceStat.pricepermeter.corr(NBHDpriceStat.greenAreaInNBHD))


# ### Change in price vs green areas

# In[56]:


# getting mean prices each year, each neighborhood
NBHDPriceYear = geo_salesNBHD.groupby(['msshchuna','Year'],as_index=False).mean('pricepermeter')


# In[57]:


# for each neighborhood, getting the diffrence between the max and min prices (max and min as year mean)
diff = NBHDPriceYear.groupby(['msshchuna']).agg({"pricepermeter": lambda x: x.max() - x.min()}).reset_index()
# renaming the "pricepermeter" column so diffrence will be known 
diff = diff.rename(columns = {'pricepermeter': 'PriceDifference'})


# In[58]:


# merging it with the green area data now we have price diffrence and sum of green area
diffGreenPrice = AreaStat.merge(diff, on = 'msshchuna')


# In[59]:


# plotting "greenAreaInNBHD" vs"PriceDifference"
# seeing the PriceDifference in each neighborhood vs the amount of green areas in it 
# plotting it with linear regression model fit.
ploti = sns.regplot(x="greenAreaInNBHD", y="PriceDifference", data=diffGreenPrice) # plotting the data
ploti.set(
    title ="Graph 9: Changes in price over the years vs amount of green areas \n in each neighborhood",
    xlabel= 'Green area normalized to neighborhood`s area',
    ylabel='Mean Price per meter max diffrence' ) # set titles 
#save fig as png 
ploti.figure.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\diffrence_price_vs_Green_area.png',
                      dpi=150 ) 


# In[60]:


# check correlation between those two series
print(diffGreenPrice.greenAreaInNBHD.corr(diffGreenPrice.PriceDifference))


# ### seems to be no real correlation betwen the two

# ## Pre Machine learning

# ### getting the data ready 

# ### Trying to improve machine learning by looking at the asset`s  floor number 

# #### Arranging the floor data - from hebrew to int

# #### Arrenging the data

# In[61]:


#### getting the floors into a separate dataframe 
floors = geo_salesNBHD.Assets_FloorNo 


# In[62]:


countfloors = floors.value_counts() # how many values are carring  each unique string in my new dataframe 
len (countfloors) # we have 385 uniqe values on this list so I wont manipulate all the data


# In[63]:


countfloors.head(10) # see what are the top values so I will work on them 


# In[64]:


# I dont want to take a risk on the data itself so I will work on a deep copy
geo_salesNums = geo_salesNBHD.copy(deep=True)


# In[65]:


# creating a converting dictionary for floors -1 to 10
floormap = {'שניה': '2', 'ראשונה': '1', 'שלישית': '3', 'רביעית': '4', 'קרקע': '0', 'חמישית': '5', 'שישית': '6', 'שביעית': '7',
'שמינית': '8', 'תשיעית': '9', 'מרתף': '-1', 'עשירית': '10','עשירית': '10'}
# using the map 
# using na_action = 'ignore' so I will only get values which I know I converted and are numbers 
geo_salesNums['Assets_FloorNo'] = geo_salesNums.Assets_FloorNo.map(floormap, na_action='ignore')


# In[66]:


geo_salesNums


# #### merging and working on the entire set of data

# In[67]:


#merging this new geo_salesNums which contains floor number as int with AreaStat
#merging the data sets green and sales with the neighborhood stat
# this is so we can see for each sale what was the green area in its NBHD along the sale details 
GreenandPrice= geo_salesNums.merge(AreaStat, on = 'msshchuna', how= 'left')


# ### Working on the data 

# #### Creating a dataframe with only he relevent columns for all of the machine learnings 

# In[68]:


GreenandPriceN= GreenandPrice[[
               'Assets_BuildingFloors',
               'Deals_YearBuilt',
               'Assets_AssetArea',
                  'greenAreaInNBHD', 
                    'Year', 
                       'pricepermeter','Assets_FloorNo']]


# ### null values

# In[69]:


#using seaborn to plot nulll values in each column
fig, ax = plt.subplots() # creating fig and subplot
fig.set_size_inches(19.7,8.27) # setting fig size 
nullplt = sns.heatmap(GreenandPriceN.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax) # creating an sns graph
nullplt.tick_params(labelsize=15) # setting labels size 
nullplt.axes.set_title( "Graph 10: Null values by column", fontsize=40) # setting a title 
nullplt.figure.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Nullplot.png', dpi=150 ) # saving


# #### we still have a lot have null values in the relevent columns so  I will drop those since I still have a lot of data 

# In[70]:


GreenandPriceN = GreenandPriceN.dropna() # reasaving 


# In[71]:


GreenandPriceN.count() # still have 17709 rows which is a lot 


# #### looking at the disturbition of the pricepermeter data

# In[72]:


plt.hist(GreenandPriceN.pricepermeter) # creating an hist graph of pricepermeter
plt.title("Graph 11: Hist of price per meter", size = 20) # set title 
plt.ylabel('Amount', size = 12) # set y label
plt.xlabel('Price Per meter', size = 12) # set x label 
plt.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\hist_pricepermeter.png', dpi=150) # save fig


# #### dropping the rare really high values 

# In[73]:


GreenandPriceN = GreenandPriceN[GreenandPriceN.pricepermeter < 150000] 
GreenandPriceN.count() # still have 17695 rows which is a lot 


# ### Save GreenPricN into a csv file

# In[74]:


GreenandPriceN.to_csv("GreenandPriceN.csv")


# ## machine learning

# ### first machine learning - with green area in the neighborhood but without assets floor number

# #### creating two sets of data

# In[264]:


X = GreenandPriceN[[
               'Assets_BuildingFloors',
               'Deals_YearBuilt',
               'Assets_AssetArea',
                  'greenAreaInNBHD', 
                    'Year' 
                    ]] # this will be used for predicting 


# In[265]:


Y = GreenandPriceN['pricepermeter'] # this we will try to predict 


# #### spliting it into train and test data those which the model will train on and those which will be used for testing how the training went

# In[266]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# #### Fit regression model

# In[267]:


lm = LinearRegression()


# In[268]:


lm.fit(X_train,y_train)


# In[269]:


# those are the coefficent values 
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Linear coef.'])
coeff_df


# #### Predicting the prices using our X_test data

# In[270]:


predicted_price = lm.predict(X_test)


# #### looking at the results

# ##### plotting the y_test data and the predicted_price to see how correct was the machine learning

# In[271]:


preforplot= pd.DataFrame({'test': y_test, 'predict':predicted_price }) # getting the data into a dataframe for sns plot


# In[272]:


fig, ax = plt.subplots() # set figure and axes 
fig.set_size_inches(15,10) # figure size 

# plotting the data and setting marker and point size 
ploti = sns.regplot(x="test", y="predict", data=preforplot,ax= ax, scatter_kws={'s':2}, marker='o', color='green') 

ploti.tick_params(labelsize=10)  # tick size
ploti.set_ylabel('Predicted Price', size = 20) # set y label and size 
ploti.set_xlabel ('Real Data', size = 20 ) # set x label and size 
# title and textsize 
ploti.axes.set_title("Graph 12: Prediction of price per meter \n with green area but without  floor number", size = 40)
#save fig as png 
ploti.figure.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Prediction_price_per_meter_withgreen.png', dpi=150 ) 


# ##### Setting x and y  limits to focous on the area where most of the data is 

# In[273]:


plt.figure(figsize=(10,10)) #  figure size 
plt.scatter(y_test, predicted_price, s=0.7, color='green') # plotting a scatter plot
plt.xlim(-1,75000) # setting x limit
plt.ylim(10000,50000) # setting y limit 
plt.title("Graph 13: Focused Prediction of price per meter \n with green area but without  floor number", size = 25) # setting a title 
plt.xlabel("Real data", size = 15) # x title 
plt.ylabel("Predicted price", size = 15) # y titile 
plt.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\focus_Prediction_price_per_meter_withgreen.png', dpi=150) # save fig


# ##### computes mean absolute error, a risk metric corresponding to the expected value of the absolute error los

# In[274]:


metrics.mean_absolute_error(y_test, predicted_price)


# #####  the coefficient of determination R^2 of the prediction.

# In[275]:


lm.score(X_test,y_test)


# ##### plotting the diffrence between predicted_price and y_test

# In[276]:


#creating the plot and saving the fig
disploti = sns.displot(predicted_price-y_test, color='green') # ploting a displot of the predicted_price minus y_test
# set title
disploti.set(title= "Graph 14: Diffrence between predicted_price and y_test \n with green area but without  floor number")
# rotate x tick labels
disploti.set_xticklabels(rotation=45)
# save 
disploti.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\diffrence_predicted_price_y_test_withgreen.png', dpi=150 ) 


# ### second machine learning - without green area in the neighborhood but with assets floor number

# #### creating two sets of data

# In[277]:


X = GreenandPriceN[[
               'Assets_BuildingFloors',
               'Deals_YearBuilt',
               'Assets_AssetArea',
                    'Year', 
                      'Assets_FloorNo']]  # these data will be used for predicting 


# In[278]:


Y = GreenandPriceN['pricepermeter'] # this we will try to predict 


# #### spliting it into train and test data those which the model will train on and those which will be used for testing how the training went

# In[279]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# #### Fit regression model

# In[280]:


lm = LinearRegression()


# In[281]:


lm.fit(X_train,y_train)


# In[282]:


# those are the coefficent values 
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Linear coef.'])
coeff_df


# #### Predicting the prices using our X_test data

# In[283]:


predicted_price = lm.predict(X_test)


# #### looking at the results

# ##### plotting the y_test data and the predicted_price to see how correct was the machine learning

# In[284]:


preforplot= pd.DataFrame({'test': y_test, 'predict':predicted_price }) # getting the data into a dataframe for sns plot


# In[285]:


fig, ax = plt.subplots() # setting ax and figure 
fig.set_size_inches(15,10) # fig size 
# plotting the data ans setting marker and point size 
ploti = sns.regplot(x="test", y="predict", data=preforplot,ax= ax, scatter_kws={'s':2}, marker='o',color='purple') 

ploti.tick_params(labelsize=10)  # tick size
ploti.set_ylabel('Predicted Price', size = 20) # set y label and size 
ploti.set_xlabel ('Real Data', size = 20 ) # set x label and size 

# title and size 
ploti.axes.set_title("Graph 15: Prediction of Price Per Meter \n Without Green Area  In the Neighborhood\n But With Assets Floor Number", size = 40)
#save fig as png 
ploti.figure.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Prediction_price_per_meter_floors.png.png', dpi=150,bbox_inches='tight' ) 


# ##### setting x and y  limits to focous on the are where most of the data is 

# In[286]:


plt.figure(figsize=(10,10)) #  fig size 
plt.scatter(y_test, predicted_price, s=0.7,color='purple') # plotting a scatter plot
plt.xlim(-1,75000) # setting x limit
plt.ylim(10000,50000) # setting y limit 
plt.title("Graph 16: Prediction of Price Per Meter \n Without Green Area In the Neighborhood \n With Assets Floor Number", size = 20) # setting a title 
plt.xlabel("Real data", size = 15 ) # x title 
plt.ylabel("Predicted price", size = 15 ) # y titile 
plt.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Focus_Prediction_price_per_meter_floors.png', dpi=150,bbox_inches='tight') # save fig


# ##### computes mean absolute error, a risk metric corresponding to the expected value of the absolute error los

# In[287]:


metrics.mean_absolute_error(y_test, predicted_price)


# #####  the coefficient of determination R^2 of the prediction.

# In[288]:


lm.score(X_test,y_test)


# ##### plotting the diffrence between predicted_price and y_test

# In[289]:


disploti = sns.displot(predicted_price-y_test, color='purple') # ploting a displot of the diffrence
# setting a title 
disploti.set(title= "Graph 17: Accuracy For Price Per Meter Prediction \n With Green Area In the Neighborhood and Assets Floor Number")
# rotatin the x tick labels 
disploti.set_xticklabels(rotation=45)
# save 
disploti.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\diffrence_predicted_price_y_test_withfloor.png', dpi=150 ) 


# ### creating a 2 dimensions numpy array 

# In[290]:


arrayX = X_train.Year.to_numpy() # converting years from series to numpy array 


# In[291]:


twoarrayX = arrayX.reshape(2, int(len(arrayX)/2)) # reshpint as metrix


# In[292]:


twoarrayX.ndim # checking how many dimentions in the array


# In[293]:


type(twoarrayX) # checking type 


# ## Third machine learning - without green area in the neighborhood and without assets` floor number

# #### creating two sets of data

# In[294]:


X = GreenandPriceN[[
               'Assets_BuildingFloors',
               'Deals_YearBuilt',
               'Assets_AssetArea',
                    'Year', 
                       ]] # those will be used for predicting 


# In[295]:


Y = GreenandPriceN['pricepermeter'] # this we will attenpt to predict


# #### spliting it into train and test data those which the model will train on and those which will be used for testing how the training went

# In[296]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# #### Fit regression model

# In[297]:


lm = LinearRegression()


# In[298]:


lm.fit(X_train,y_train)


# In[299]:


# those are the coefficent values 
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Linear coef.'])
coeff_df


# #### Predicting the prices using our X_test data

# In[300]:


predicted_price = lm.predict(X_test)


# #### looking at the results

# ##### plotting the y_test data and the predicted_price to see how correct was the machine learning

# In[301]:


preforplot= pd.DataFrame({'test': y_test, 'predict':predicted_price }) # getting the data into a dataframe for sns plot


# In[302]:


fig, ax = plt.subplots() # setting figure and axes 
fig.set_size_inches(15,10) # figure size 
# plotting the data ans setting marker and point size 
ploti = sns.regplot(x="test", y="predict", data=preforplot,ax= ax, scatter_kws={'s':2}, marker='o', color='red') 
ploti.tick_params(labelsize=10)  # tick size
ploti.set_ylabel('Predicted Price', size = 20) # set y label and size 
ploti.set_xlabel ('Real Data', size = 20 ) # set x label and size 
# title and size 
ploti.axes.set_title("Graph 18: Prediction of Price Per meter \n Without Green Area In the Neighborhood \n Without Assets` Floor Number", size = 40)
#save fig as png 
ploti.figure.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\Prediction_price_per_meter.png', dpi=150 ,bbox_inches='tight') 


# ##### setting x and y  limits to focous on the majority of the data 

# In[303]:


plt.figure(figsize=(10,10)) #  figure size 
plt.scatter(y_test, predicted_price, s=0.7, color='red') # plotting a scatter plot
plt.xlim(-1,75000) # setting x limit
plt.ylim(10000,50000) # setting y limit 
plt.title("Graph 19: Focoused Prediction of Price Per Meter \n Without Green Area in the Neighborhood \n Without Assets` Floor Number", size = 20) # setting a title 
plt.xlabel("Real Data", size = 15) # x title 
plt.ylabel("Predicted Price", size = 15 ) # y titile 
# saving
plt.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\focus_Prediction_price_per_meter.png', dpi=150,bbox_inches='tight') # save fig


# ##### computes mean absolute error, a risk metric corresponding to the expected value of the absolute error los

# In[304]:


metrics.mean_absolute_error(y_test, predicted_price)


# #####  the coefficient of determination R^2 of the prediction.

# In[305]:


lm.score(X_test,y_test)


# ##### plotting the diffrence between predicted_price and y_test

# In[306]:


disploti = sns.displot(predicted_price-y_test, color='red') # plotting a displot of diffrence between predited and real data
# set title 
disploti.set(title= "Graph 20: Diffrence Between Predicted Price and y_test \n Without Green Area In the Neighborhood \n Without Assets` Floor Number")
# rotating x tick labels 
disploti.set_xticklabels(rotation=45)
# saving
disploti.savefig(r'C:\Users\doron\OneDrive\שולחן העבודה\Uni\final_ex_ds\outputs\diffrence_predicted_price_y_test.png', dpi=150 ) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




