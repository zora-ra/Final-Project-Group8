import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
##----------------------------Data preprocessing-------------------------------
# Load train and test data
test_data = pd.read_csv('exoTest.csv')
train_data = pd.read_csv('exoTrain.csv')

# Check Shape and Dimension
test_data.shape
train_data.shape

# Take a look at the first few rows
test_data.head()
train_data.head()

# Check for Missing Values/ Remove missing values

# Total missing for each feature
train_data.isnull().sum()
# Any missing values at all?
train_data.isnull().values.any()
# Total count of missing values
train_data.isnull().sum().sum() # There is no missing vaules, no need to remove

#split X, y, X_test, y_test
X = train_data.iloc[:, 1:]
y = train_data.iloc[:,:1].values

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:,:1].values

#Smooth data
for i in X.columns:
    X[i] = (30*X[i] + uniform_filter1d(X[i],size = 100))/31
X = X.values

#train, validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling...
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_val = sc_y.fit_transform(y_val)


