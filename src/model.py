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
test = pd.read_csv('276.22372_.csv')

print(train['no_likes'].mean())
print(train.describe())
#print(train.head(20))
print(test['no_likes'].mean())

test['no_likes'] = test['no_likes'] + 13
#print(test.head(5))
print(test[(test['no_likes']>300)])

print(test[(test['no_likes']<55)])
print(test['no_likes'].mean())
print(test['no_likes'].describe())
test.to_csv('submission2.csv',index=False)
# creating date for analysis
train['date'] = 1000*train['month'] + 100*train['weekday'] + train['hour']
#test['date'] = 1000*test['month'] + 100*test['weekday'] + test['hour']
train['page_likes/date'] =  train['page_total_likes'] / train['date']
#test['page_likes/date'] =  test['page_total_likes'] / test['date']

fig, ax = plt.subplots()
ax.scatter(x = train['date'], y = train['no_likes'])
plt.xlabel('date', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['month'], y = train['hour'])
plt.xlabel('month', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['date'], y = train['page_total_likes'])
plt.xlabel('date', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['page_likes/date'], y = train['no_likes'])
plt.xlabel('page_likes/date', fontsize=13)
plt.show()

train.groupby('category')['category'].count().plot(kind='bar')
plt.show()
# Category 1 with some more
train.groupby('type_of_post')['type_of_post'].count().plot(kind='bar')
plt.show()
# Photos are a lot more
train.groupby('weekday')['weekday'].count().plot(kind='bar')
plt.show()
# Nearly the same
train.groupby('hour')['hour'].count().plot(kind='bar')
plt.show()
# Hour 3 and 10 with a lot more
train.groupby('month')['month'].count().plot(kind='bar')
plt.show()
# Month 10, 7, 12 and 6 

#######################################################################################################

#Visualizing 'outlier' suspects 
print("Suspect outliers:\n", train[(train['hour']>13) 
                                    | (train['no_likes']>3000) 
                                    | (train['month']<5)] )
print("\n")
#Deleting outliers
train = train.drop(train[(train['no_likes']>3000) 
                        | (train['month']<5)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(x = train['date'], y = train['no_likes'])
plt.xlabel('date', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['page_likes/date'], y = train['no_likes'])
plt.xlabel('page_likes/date', fontsize=13)
plt.show()


print(train['no_likes'].mean())
print(train['no_likes'].describe())
all_data = pd.concat((train, test)).reset_index(drop=True)
print("TOTAL MEAN  ",all_data['no_likes'].mean())
#test_ID = test['ID']

sub = pd.DataFrame()
sub['ID'] = test_ID
sub['no_likes'] = [train['no_likes'].mean()]*150


