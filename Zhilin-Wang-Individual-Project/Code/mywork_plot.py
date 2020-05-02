##----------------------------  Load packages  -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

##----------------------------Data preprocessing-------------------------------
# Load train and test data
test_data = pd.read_csv('exoTest.csv')
train_data = pd.read_csv('exoTrain.csv')
# Check Shape and Dimension
print(test_data.shape)
print(train_data.shape)

# Take a look at the first few rows
test_data.head()
train_data.head()

# check the distribution of label
print(train_data.LABEL.value_counts())
print(test_data.LABEL.value_counts())

# encoding the label (1 to 0, 2 to 1)
train_data['LABEL'] = train_data['LABEL'].replace([1], [0])
train_data['LABEL'] = train_data['LABEL'].replace([2], [1])
test_data['LABEL'] = test_data['LABEL'].replace([1], [0])
test_data['LABEL'] = test_data['LABEL'].replace([2], [1])

# Check for Missing Values/ Remove missing values
# Total missing for each feature
train_data.isnull().sum()
# Any missing values at all?
train_data.isnull().values.any()
# Total count of missing values
train_data.isnull().sum().sum() # There is no missing vaules, no need to remove

# split X, y, X_test, y_test
X = train_data.iloc[:, 1:]
y = train_data.iloc[:,:1]

X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:,:1]

#print the shape of y_test
print(y_test.shape)

# remove upper outliers
# Since we are looking for dips in flux when exo-planets pass between the telescope and the star,
# we should remove any upper outliers.
def reduce_upper_outliers(df, reduce=0.01, half_width=4):
    length = len(df.iloc[0, :])
    remove = int(length * reduce)
    for i in df.index.values:
        values = df.loc[i, :]
        sorted_values = values.sort_values(ascending=False)
        # print(sorted_values[:30])
        for j in range(remove):
            idx = sorted_values.index[j]
            # print(idx)
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            # print(idx,idx_num)
            for k in range(2 * half_width + 1):
                idx2 = idx_num + k - half_width
                if idx2 < 1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.' + str(idx2)]  # corrected from 'FLUX-' to 'FLUX.'

                count += 1
            new_val /= count  # count will always be positive here
            # print(new_val)
            if new_val < values[idx]:  # just in case there's a few persistently high adjacent values
                df.iloc[i][idx] = new_val
    return df
# reference: https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration

X = reduce_upper_outliers(X)
X_test = reduce_upper_outliers(X_test)

#get the dataset after removing outliers
td = pd.concat([X,y],axis = 1)

#method to plot exoplanet
def p(train_data,i):
    flux = train_data[train_data.LABEL == 1].drop('LABEL', axis=1).iloc[i, :]
    time = np.arange(len(flux)) * (36.0 / 60.0)
    plt.plot(time, flux)
#method to plot the non exoplanet
def pp(train_data,j):
    flux = train_data[train_data.LABEL == 0].drop('LABEL', axis=1).iloc[j, :]
    time = np.arange(len(flux)) * (36.0 / 60.0)
    plt.plot(time,flux)

# Smooth data
def smoonthdata(X):
    for i in range(len(X)):
        X.iloc[i] = uniform_filter1d(X.iloc[i], size=100)
    return X

X = smoonthdata(X)

#get the dataset after smoothing the data
ttd = pd.concat([pd.DataFrame(X),y],axis = 1)
X = X.values
#scaling
X = ((X - np.mean(X, axis=1).reshape(-1,1)) / np.std(X, axis=1).reshape(-1,1))[:,:,np.newaxis]
print(X.shape)
#----------------------------------------------Data preprocessing plot--------------------------------------------------
#plot the exoplanet
for i in[3,9,15]:
     plt.figure(figsize=(15,5))
     plt.title('Flux of star {} (confirmed exoplanet)'. format(i+1))
     plt.ylabel('Flux, e-/s')
     plt.xlabel('Time, hours')
     p(train_data,i)
     p(td,i)
     p(ttd,i)
     plt.legend(['Original', 'Outlier removed','smoothed'], loc='upper left')
     plt.show()
#plot the non exoplanet
for j in [999, 2999, 3999]:
    plt.figure(figsize=(15, 5))
    plt.title('Flux of star {} (non-confirmed exoplanet)'.format(j + 1))
    plt.ylabel('Flux, e-/s')
    plt.xlabel('Time, hours')
    pp(train_data, j)
    pp(td, j)
    pp(ttd, j)
    plt.legend(['Original', 'Outlier removed', 'smoothed'], loc='upper left')
    plt.show()
