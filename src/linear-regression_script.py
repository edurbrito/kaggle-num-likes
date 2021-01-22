#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics

#Limiting floats output to 5 decimal points
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x)) 

#Now let's import and put the train and test datasets in  pandas dataframe
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##display the first five rows of the train dataset.
print(train.head(5))
print("\n")

##display the first five rows of the test dataset.
print(test.head(5))
print("\n")

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))
#Save the 'Id' column
train_ID = train['ID']
test_ID = test['ID']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("ID", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)


# creating date for analysis
train['date'] = 1000*train['month'] + 100*train['weekday'] + train['hour']
test['date'] = 1000*test['month'] + 100*test['weekday'] + test['hour']
train['page_likes/date'] =  train['page_total_likes'] / train['date']
test['page_likes/date'] =  test['page_total_likes'] / test['date']

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
print("\n")
  
fig, ax = plt.subplots()
ax.scatter(x = train['date'], y = train['no_likes'])
plt.xlabel('date', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['date'], y = train['page_total_likes'])
plt.xlabel('date', fontsize=13)
plt.xlim(12000,13000)
plt.ylim(135000,140000)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['page_likes/date'], y = train['no_likes'])
plt.xlabel('page_likes/date', fontsize=13)
plt.show()

train.groupby('category')['category'].count().plot(kind='bar')
plt.show()
# # Category 1 with some more
train.groupby('type_of_post')['type_of_post'].count().plot(kind='bar')
plt.show()
# # Photos are a lot more
train.groupby('weekday')['weekday'].count().plot(kind='bar')
plt.show()
# # Nearly the same
train.groupby('hour')['hour'].count().plot(kind='bar')
plt.show()
# # Hour 3 and 10 with a lot more
train.groupby('month')['month'].count().plot(kind='bar')
plt.show()
# # Month 10, 7, 12 and 6 
fig, ax = plt.subplots()
ax.scatter(x = train['hour'], y = train['no_likes'])
plt.xlabel('hour', fontsize=13)
plt.ylabel('no_likes', fontsize=13)
plt.show()
#######################################################################################################

#Visualizing 'outlier' suspects 
print("Suspect outliers:\n", train[(train['hour']>13) 
                                    | (train['no_likes']>3000) 
                                    | (train['month']<5)] )
print("\n")
#Deleting outliers
train = train.drop(train[(train['hour']>20) 
                        | (train['no_likes']>3000) 
                        | (train['month']<5)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(x = train['hour'], y = train['no_likes'])
plt.xlabel('hour', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['page_likes/date'], y = train['no_likes'])
plt.xlabel('page_likes/date', fontsize=13)
plt.show()

#######################################################################################################

print(train['no_likes'].describe())

sns.distplot(train['no_likes'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['no_likes'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('no_likes distribution')
plt.show()

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['no_likes'], plot=plt)
plt.show()

################################

#We use the numpy fuction log1p which  applies log(x+1) to all elements of the column
train["no_likes"] = np.log2(train["no_likes"]+1) 

#Check the new distribution 
sns.distplot(train['no_likes'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['no_likes'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('no_likes distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['no_likes'], plot=plt)
plt.show()


#######################################################################################################

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.no_likes.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['no_likes'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

#Correlation map to see how features are correlated with no_likes
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

#######################################################################################################

# As string
all_data['type_of_post'] = all_data['type_of_post'].apply(str)
#Changing category into a categorical variable
all_data['type_of_post'] = all_data['type_of_post'].astype(str)
all_data['category'] = all_data['category'].astype(str)
all_data['month'] = all_data['month'].astype(str)
all_data['weekday'] = all_data['weekday'].astype(str)
all_data['hour'] = all_data['hour'].astype(str)
all_data['paid'] = all_data['paid'].astype(str)

fig, ax = plt.subplots()
ax.scatter(x = all_data['date'], y = all_data['page_total_likes'])
plt.xlabel('date', fontsize=13)
plt.show()

#######################################################################################################

from sklearn.preprocessing import LabelEncoder
cols = ("type_of_post","paid")
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
all_data.drop(['page_total_likes','month','weekday','hour','date'], axis=1, inplace=True)
train = all_data[:ntrain]
test = all_data[ntrain:]

#print(train.tail(5))
#print(test.head(5))

corrmat = train.corr()
plt.subplots()
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

corrmat = test.corr()
plt.subplots()
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

#######################################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(train.values, y_train, test_size=0.3,random_state=1) 

# create linear regression object 
reg = LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, Y_train) 

# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_train, Y_train)))

# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
plt.xlim(-2,20)
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - Y_train, 
            color = "green", s = 10, label = 'Train data') 
  
# ## plotting residual errors in test data 
# plt.scatter(reg.predict(X_test), reg.predict(X_test) - Y_test, 
#             color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 

# model = LinearRegression().fit(train.values, y_train)
# stacked_train_pred = model.predict(train.values)
# stacked_pred = np.exp2(model.predict(test.values))-1
# print("Coef: ",model.coef_)
# print("RMSE(Train): ", rmsle(stacked_train_pred,y_train ))
# print(rmsle([-1]*150, stacked_pred)) # As RapidMiner


# train['no_likes'] = np.exp2(y_train)-1
# train['no_likes_pred'] = np.exp2(stacked_train_pred)-1
# print(train.head(20))

# sub = pd.DataFrame()
# sub['ID'] = test_ID
# sub['no_likes'] = stacked_pred

# #print(sub.describe())
# sub.to_csv('submission2.csv',index=False)