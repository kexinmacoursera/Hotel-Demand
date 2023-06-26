# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from calendar import month_name
from sklearn.preprocessing import LabelEncoder, StandardScaler 

#Step 1
#Loading hotel booking data set

DF = pd.read_csv("C:/Users/limjo/Desktop/hotel_bookings.csv")

#Step 2
#Data exloration and Data Cleaning

pd.options.display.max_columns = None
pd.options.display.max_rows = None

print(DF.columns.values)

# Display the first ten rows
print(DF.head(10))

#Check the size of the date set

print(DF.shape)

#Total there are 120k rows and 32 columns

DF.info()

#We noticed some missing values here, column Agent, Company and Country
#Let's get a closer look, what's the percentage of the missing values

print("Count of Missing Values", DF.isna().sum()/ len(DF.company))

#94.3% of the missing values, I decided to drop the column
#Once drop the column and I will remove the missing values in country and agent as the impact is much lower (0.004% and 0.13%)
DF = DF.drop(['company'], axis = 1)

# The rest of the columns that contain missing values are not significant, I can either drop them or replace missing values with our assumptions:
    
# Agent, If there's no agent I could assume the customers book the hotel directly.
# Children, If there's N/A, I would assume is an 0. It's common that only adutls travel and stay in hotels. I miss that moment though
# Company, will drop the value.

Null_Rplc = {"children":0, "agent":0}

DF_cleaned = DF.fillna(Null_Rplc)

DF = DF.dropna(axis = 0)

#Let's check again!

DF.info() #Perfect! All missing values either removed or imputed with different values.

#What are the unique values in each column?
for col in DF.describe(include = 'object').columns:
    print(col)
    print(DF[col].unique())
    
#The data looks very consistece now but I would prefert to have the country name instead of the abbreviation, It would be better to leverage 
#external data to lookup the value instead of replace manually

# Bring in the external data

url = 'https://www.iban.com/country-codes'
html = requests.get(url).content
countrycode_list = pd.read_html(html)

countrycode_list_df= countrycode_list[-1]


DF_Merged = pd.merge(left=DF, right =countrycode_list_df,how = 'left', left_on='country', right_on = 'Alpha-3 code' )


#Outliers, first will sorted out all the numeric vaeriables then will do a boxplot 
DF_Merged.describe()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics_col = DF_Merged.select_dtypes(include =numerics)

numerics_col_only = numerics_col.drop(columns = ['is_canceled',
                                 'arrival_date_year',
                                 'arrival_date_week_number',
                                 'arrival_date_day_of_month',
                                 'is_repeated_guest',
                                 'previous_cancellations',
                                 'previous_bookings_not_canceled',
                                 'booking_changes',
                                 'days_in_waiting_list',
                                 'required_car_parking_spaces',
                                 'total_of_special_requests',
                                 ])

# Boxplot
# https://www.kaggle.com/code/marta99/hotel-booking-demand-eda-visualization
#why use stripplot?    
n=1
sns.set_style('darkgrid')
sns.set(font_scale = 0.9)
plt.figure(figsize = (10, 17))

for numerics_column in numerics_col_only:
    plt.subplot(6,3,n)
    sns.stripplot(x = DF_Merged['hotel'], y = DF_Merged[numerics_column], palette = 'Paired').set(xlabel = None)
    hue = ''
    plt.title(f'{numerics_column} boxplot')
    n = n + 1
    plt.tight_layout()

DF.hist(figsize=(22,16))
plt.show()
# We clearly see an outlier in adr column, Resort more than 25 ppl but it is possible, even though its possible more than


# IQR method:
    
# Calculate the upper and lower limits
Q1 = DF_Merged['adr'].quantile(0.25)
Q3 = DF_Merged['adr'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

# Create arrays of Boolean values indicating the outlier rows
upper_array = np.where(DF_Merged['adr']>=upper)[0]
lower_array = np.where(DF_Merged['adr']<=lower)[0]

 
# Removing the outliers
# Method 1 Removing outliers based on IQR
DF_Merged.drop(index=upper_array, inplace=True)
DF_Merged.drop(index=lower_array, inplace=True)
 

# Method 2 Removing outliers based on fix value

DF_Merged = DF_Merged[DF_Merged['babies']<8]

# Create a family column:
DF_Merged['kids'] = DF_Merged['children'] + DF_Merged['babies']
DF_Merged['family'] = np.where(DF_Merged['kids']>0, '1','0')


describe_df = DF_Merged.describe()

describe_df.to_csv('describe_df.csv')
#Write to csv

#DF_Merged.to_csv('newdata.csv', index = False)

''' Check Outliear again, it has been removed!

n=1
sns.set_style('darkgrid')
sns.set(font_scale = 0.9)
plt.figure(figsize = (10, 17))

for numerics_column in numerics_col_only:
    plt.subplot(6,3,n)
    sns.stripplot(x = DF_Merged['hotel'], y = DF_Merged[numerics_column], palette = 'Paired').set(xlabel = None)
    hue = ''
    plt.title(f'{numerics_column} boxplot')
    n = n + 1
    plt.tight_layout()

'''

#EDA:
#Hotel categories:
plt.figure(figsize=(5,6))
plt.pie(DF_Merged.hotel.value_counts(),explode = [0,0.15],labels = DF_Merged.hotel.value_counts().index, colors=sns.color_palette('Set1'), shadow= True,autopct='%1.1f%%')
plt.legend()
plt.title('Type of Hotels Customer Booked')
plt.show()


    
#Prefereed meal plan

'''DF_Merged['Meal'] = DF_Merged['meal'].map({'BB': 'Bed & Breakfast',
                                           'HB': 'Half board',
                                           'FB': 'Full board',
                                           'FC': 'no meal package',
                                           'Undefined': 'no meal package'})
'''


meal_labels= ['Bed & Breakfast','Half board', 'No Meal', 'No Meal', 'Full board']
size = DF_Merged['meal'].value_counts()
plt.figure(figsize=(6,6))
cmap =plt.get_cmap("Pastel2")
colors = cmap(np.arange(3)*4)
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(size, labels=meal_labels, colors=colors, wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Meal Types', weight='bold')
plt.show()

Meal_Ratio= pd.DataFrame(DF_Merged.meal.value_counts(normalize=True)*100)
print(Meal_Ratio)



# Conclusion: We can tell City hotels has more booking than Resort
# Roughly 70% of the population booked the City hotel

# New guest or repeated guest?  A:Only a few bookings were made by repeated customers.
# Hotels are difnitely experiencing low rates of repeated guests. It is important to forcus on improving
# guest loyalty and increasing the likelyhood of guests returning for future stays.

'''
Some strategies to consider
1. Loyalty program: Implement a robust loyalty program that rewards guests to their repeat stays.
Offer incentives such as: exclusive discounts, free room upgrades? Or points that can be redeemed for future stays.
2. Partnerships with Local Businesses: Collaborate with local businesses, such as restaurants, attractions, 
or event organizers, to offer joint promotions or packages for guests. 
This can provide added value to their stay and encourage them to return to experience the local area.
    
'''

plt.figure(figsize=(5,6))
plt.pie(DF_Merged.is_repeated_guest.value_counts(),labels = DF_Merged.is_repeated_guest.value_counts().index, colors=sns.color_palette('Set2'), shadow= True,autopct='%1.1f%%')
plt.legend()
plt.title('New guest or repeated guest?')
plt.show()


# Reservation status in different hotels


plt.figure(figsize = (8,4))
ax1= sns.countplot(x = 'hotel', hue = 'is_canceled',data = DF_Merged, palette = 'YlOrRd')
legend_labels,_ = ax1. get_legend_handles_labels()
plt.title('Reservation status in different hotels',size = 12)
for container in ax1.containers:
    ax1.bar_label(container)
plt.xlabel('Hotel Category')
plt.ylabel('Number of Cancellation')

#Total Cancellation

Total_perc = DF_Merged['is_canceled'].value_counts(normalize = True)

print("Total Cancellation rate is "
      ,f"{(Total_perc).iloc[1]:.2%}")

#Percentage cancelled
Resort_perc = DF_Merged[DF_Merged['hotel'] == 'Resort Hotel']
R_Perc= Resort_perc['is_canceled'].value_counts(normalize= True)
print("Resort Cancellation rate is "
      f"{(R_Perc).iloc[1]:.2%}")

City_hotel_perc = DF_Merged[DF_Merged['hotel'] == 'City Hotel']
C_Perc = City_hotel_perc['is_canceled'].value_counts(normalize= True)

print("City Cancellation rate is "
      f"{(C_Perc).iloc[1]:.2%}")

# Apparently Resort cancellation rate is lower than City Hotel but why?
# Resort hotels have a lower canceellation rate. This makes sense, since a Resort hotel is usually for beneficial purposes,
# while a City hotel is a business-type or practcial place to nest. 
# It can be assumed that a resort hotel is part of a larger trip plan, and therefore more stable

# Cancellation relationship with Deposit type


plt.figure(figsize = (8,4))
ax2= sns.countplot(x = 'deposit_type', hue = 'is_canceled',data = DF_Merged, palette = 'YlOrRd')
legend_labels,_ = ax1. get_legend_handles_labels()
plt.title('Deposit Type of Cancelled Bookings',size = 12)
for container in ax2.containers:
    ax2.bar_label(container)
plt.xlabel('Deposit Type')
plt.ylabel('Number of Cancellation')

# We noticed that Most of the cancelled reservations did not require a deposit!
# Even though Resort cancellation rate is lower than City Hotel, but still both type of hotels are being affected
# by a high rate of cancellations. The fact that a large number of cancellations did not have a deposit. 
# Suggestion: Company should make the cancellation more restrictive. Or flexiable cancellation options, such as free cancellation
# up to a certain time or a reduced cancellation fee, hotels can incentivize guests to keep their reservations.
# Hotel could conduct a brief survey of customers when they cancel a reservation to find out the reason behind it.

# Cancellation relationship with Market Segment
plt.figure(figsize = (8,4))
ax3= sns.countplot(x = 'country', hue = 'is_canceled',data = DF_Merged, palette = 'YlOrRd')
legend_labels,_ = ax3. get_legend_handles_labels()
plt.title('Cancellations by distribution channelt',size = 12)
for container in ax3.containers:
    ax3.bar_label(container)
    
plt.xlabel('country')
plt.ylabel('Number of Cancellation')

# Many cancellation were coming from Online TA and Offline TA/TO, how's the impact?

Segment_perc = DF_Merged[(DF_Merged.market_segment == 'Online TA') 
                         |(DF_Merged.market_segment == 'Offline TA/TO')
                         |(DF_Merged.market_segment == 'Groups')
                         ]
Total_canceled = len(DF_Merged[DF_Merged["is_canceled" ] == 1])


Segment_Canceled = len(Segment_perc[Segment_perc["is_canceled" ] == 1])

print(Segment_Canceled/ Total_canceled)

# Online TA and Offline TA/TO made up 70% of the cancellation 

# Do higher prices make guests more likely to cancel their reservation? 


ax4 = sns.boxplot(x= "reservation_status",y="adr", hue = 'hotel',data= DF_Merged,
            palette="YlOrRd").set(title = 'Distribution of ADR vs reservation status',
                                xlabel='Hotel Type', ylabel='Average Daily Rate ')
sns.move_legend(ax4, 'lower right')

# Prices seemed to have a small effect on whether people would cancel or not show up for the city hotel. The effect was more pronounced for the resort, lower prices may have led to more people not showing up.
#https://rpubs.com/jiajianwoo95/706496


request_GP = DF_Merged.groupby([ 'total_of_special_requests', 'is_canceled']).size().unstack(fill_value=0)
ax7 = request_GP.plot(kind='bar', stacked=True, cmap='vlag', figsize=(10,10),  color=['g','b'])
plt.title('Total Special Request vs Booking Cancellation Status', weight='bold')
plt.xlabel('Number of Special Request', fontsize=12)
plt.xticks(rotation=360)
plt.ylabel('Count', fontsize=12)
for bars in ax7.containers:
    ax7.bar_label(bars)





# Reservation status in different months

'''plt.figure(figsize = (8,4))
ax1= sns.countplot(x = 'arrival_date_month', hue = 'is_canceled',data = DF_Merged, palette = 'YlOrRd')
legend_labels,_ = ax1. get_legend_handles_labels()
plt.title('Reservation status in different hotels',size = 20)
plt.xlabel('Hotel Category')
plt.ylabel('number of reservations')'''

# Market Segment on spending
plt.figure(figsize =(10,5))
ax5 = sns.barplot(data = DF_Merged, x = "market_segment", y = "adr")
plt.legend(fontsize = 4)
plt.xlabel('Market_segment',fontsize = 8)
plt.ylabel('Dollar Spending',fontsize = 10)
plt.title('Spending by Marketing Segment',fontsize = 16)
for i in ax5.containers:
    ax5.bar_label(i,)

# Where the customer bookings coming from?

ax7= sns.countplot(x='market_segment',data=DF_Merged,ec='black',lw=1)
plt.title("Popular Market Segments")
plt.xticks(rotation=90)
plt.show()



#Conclustion:Only 6% of the customer were coming from Direct booking, however, the customer tend the spend highest amount
#Suggestion: Hotel should encourage more direct booking by offering coupons 
#Examples:
#1. Direct Booking Incentives: Offer additional incentives for customers who book directly
#such as free Wi-Fi, parking, welcome drinks, or other exclusive amenities. 
#These perks add value to the booking experience and make direct booking more attractive.
#2. Email marketing, and social media engagement.

#How does the hotel rate varies over the year.

Monthly_price = DF_Merged[["hotel","arrival_date_month","adr"]].sort_values("arrival_date_month")

MIO = ["January", "February", "March", "April", "May", "June", 
       "July", "August", "September", "October", "November", "December"]

Monthly_price["arrival_date_month"]  = pd.Categorical(Monthly_price["arrival_date_month"], categories=MIO, ordered=True)



# Hotest season:

#Create two dataframes to seprate hotel type:
    
DF_Resort = DF_Merged[(DF_Merged['hotel']== 'Resort Hotel') & (DF_Merged['is_canceled']==0)]


DF_Hotel =  DF_Merged[(DF_Merged['hotel']== 'City Hotel') & (DF_Merged['is_canceled']==0)]


resort_visitors = DF_Resort['arrival_date_month'].value_counts().reset_index()
resort_visitors.columns = ['Month','Resort Number of Visitors']

hotel_visitors = DF_Hotel['arrival_date_month'].value_counts().reset_index()
hotel_visitors.columns = ['Month','City Hotel Number of Visitors']

Total_visitors = resort_visitors.merge(hotel_visitors, on='Month')

#Sort by Month
Total_visitors_sorted = Total_visitors.sort_values('Month', key= lambda x :pd.Categorical(x,categories= MIO, ordered= True))
plt.figure(figsize = (17, 8))
sns.set_style('ticks')
sns.lineplot(data= Total_visitors_sorted,x = "Month", y="City Hotel Number of Visitors",label = 'City Hotel',marker='o',color='r')
sns.lineplot(data= Total_visitors_sorted,x = "Month", y="Resort Number of Visitors",label = 'Resort Hotel', marker = 'D', color='g')
plt.title("Number of Visitors per month", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of Vistors", fontsize=16)
plt.grid(True)
plt.show()



#Barplot with STD

plt.figure(figsize=(17,8))
sns.lineplot(x = "arrival_date_month", y="adr", hue="hotel", palette=['r', 'g'],data=DF_Merged, 
            hue_order = ["City Hotel", "Resort Hotel"], size="hotel", sizes=(2.5, 2.5))
plt.title("Room price per night over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Hotel Rate [EUR]", fontsize=16)
plt.grid(True)
plt.show()

''''
Monthly_price2 = DF_Merged[["hotel","arrival_date_month","adr","revenue"]].sort_values("arrival_date_month")

MIO = ["January", "February", "March", "April", "May", "June", 
       "July", "August", "September", "October", "November", "December"]

Monthly_price2["arrival_date_month"]  = pd.Categorical(Monthly_price2["arrival_date_month"], categories=MIO, ordered=True)


plt.figure(figsize=(17,8))
sns.lineplot(x = "arrival_date_month", y="revenue", palette=['r', 'g'],data=Monthly_price2, 
            hue_order = ["City Hotel", "Resort Hotel"], size="hotel", sizes=(2.5, 2.5))
plt.title("Room price per night over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Hotel Rate [EUR]", fontsize=16)
plt.grid(True)
plt.show()
'''

# City Hotels has overall more visitors compare to Resort hotel. City hotels visitors increase dramatically from March until end of summer
# Resort Hotels does not have that much of flatucation and relatively flat, however if we go back to the pricing the visitors spend
# The graph clearly shows that the prices in the resort hotels are much higher during the summer
# The price of the city hotel varies less than Resort hotel.

#Between May and August attendance is at its peak, it is the high season. Seasonal downturns are experienced by every
#one involved in the journey. They have a significant impact on hotel industries revenues. However, the hotel owner
#should not remain passive but take advantage of this moment to establish a long-term strategy to mitigate the 
#consequences of seasonality on its turnover.
#https://www.amenitiz.com/en/blog/the-effects-of-seasonality-on-the-hotel-industry/?p_a=false
#https://www.ezeeabsolute.com/blog/increase-low-season-hotel-occupancy/

# City hotel peak peak month in August and Resort hotel peak month in July

 


# How many days customer would like to stay?

DF_Merged['Total Night stays'] = DF_Merged['stays_in_weekend_nights']+ DF_Merged['stays_in_week_nights']

plt.figure(figsize=(20,18))
sns.boxplot(x = "Total Night stays", y = "market_segment", data = DF_Merged, hue = "hotel", orient="h",palette = 'light:#5A9', fliersize = 10)
plt.xlabel( "Total night Stays" , size = 20 )
plt.ylabel( "Market Segment" , size = 20)

# Are hotels family friendly?



plt.figure(figsize = (8,4))
ax2= sns.countplot(x = 'hotel', hue = 'family',data = DF_Merged, palette = 'YlOrRd')
legend_labels,_ = ax1. get_legend_handles_labels()
plt.title('',size = 20)
for container in ax2.containers:
    ax2.bar_label(container)
plt.xlabel('Type of hotel')
plt.ylabel('Families by Hotel')


#the low occupancy of rooms by adults with children seems to indicate that none of the hotels has a family-friendly 
#atmosphere. Consequently, it would be worth considering what services could be included to attract this type of client 
#or whether, on the contrary, efforts should be focused on attracting couples and groups of adults to the hotel, 
#offering more exclusive services.


DF_Merged['is_canceled_new'] = DF_Merged['is_canceled'].replace(['0', '1'], ['1', '-1'])

DF_Merged['is_canceled'] = DF_Merged['is_canceled'].replace([0,1], [1,-1])

DF_Merged['revenue'] = DF_Merged["Total Night stays"]*DF_Merged['adr']*DF_Merged['is_canceled']
DF_Merged['revenue'].groupby(DF_Merged.is_canceled).sum()


sns.scatterplot(data=DF_Merged, x=DF_Merged.index, y='is_canceled', hue='lead_time')





#ML part

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Hotel_data = DF_Merged.copy()

list_1=list(Hotel_data.columns)

list_cate=[]
for i in list_1:
    if Hotel_data[i].dtype=='object':
        list_cate.append(i)

for i in list_cate:
    Hotel_data[i]=le.fit_transform(Hotel_data[i])

y=Hotel_data['hotel']
x=Hotel_data.drop('hotel',axis=1)


Hotel_data.drop(['Alpha-3 code', 'Alpha-2 code','Country','Numeric','country','previous_cancellations','reservation_status'], axis=1,inplace = True)

Hotel_data.corr()
correlations = Hotel_data.corr()["is_canceled"].sort_values()

#
sns.violinplot(data = DF_Merged,x='is_canceled', y='lead_time')




sns.pairplot(Hotel_data)




from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

le=LabelEncoder()

Hotel_data = DF_Merged.copy()
Hotel_data= Hotel_data.dropna()

Hotel_data.info()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
score_1=accuracy_score(y_test,pred_1)

score_1


sns.relplot(x='is_canceled', y = 'deposit_type', data = Hotel_data)