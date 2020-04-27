import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import SGD
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

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


# take a look at the shape of features
# for i in[0, 9, 14, 19, 24, 29]:
#     flux = train_data[train_data.LABEL==2].drop('LABEL', axis=1).iloc[i,:]
#     time = np.arange(len(flux))*(36.0/60.0)
#     # the sampling frequency is 1 / (36 minutes * 60 seconds in a minute) or 0.00046 Hz
#     plt.figure(figsize=(15,5))
#     plt.title('Flux of star {} with confirmed exoplanets'. format(i+1))
#     plt.ylabel('Flux, e-/s')
#     plt.xlabel('Time, hours')
#     plt.plot(time,flux)
#     plt.show()
#
# for j in [0, 999, 1999, 2999, 3999, 4999]:
#     flux = train_data[train_data.LABEL==1].drop('LABEL', axis=1).iloc[j,:]
#     time = np.arange(len(flux))*(36.0/60.0)
#     plt.figure(figsize=(15,5))
#     plt.title('Flux of star {} with NO confirmed exoplanets'. format(j+1))
#     plt.ylabel('Flux, e-/s')
#     plt.xlabel('Time, hours')
#     plt.plot(time,flux)
#     plt.show()

# conclusion:
# there are many shapes and distributions in the feature data, the data of no confirmed exo-planets star is more flat,
# the data of confirmed exo-planets star is more fluctuated. So we need to clean the data. We will remove outliers,
# smooth data and scale features.

# split X, y, X_test, y_test
X = train_data.iloc[:, 1:]
y = train_data.iloc[:,:1]
# print(y.shape)
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:,:1]


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

# Smooth data
for i in X.columns:
    X[i] = (30*X[i] + uniform_filter1d(X[i], size=100))/31
X = X.values

for i in X_test.columns:
    X_test[i] = (30*X_test[i] + uniform_filter1d(X_test[i], size=100))/31
X_test = X_test.values

#train, validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2,  random_state = 0)
from sklearn.model_selection import KFold


# balance the data
# generate more data using rotation
x_train_positives = X_train[np.squeeze(y_train) == 1]
x_train_negatives = X_train[np.squeeze(y_train) == 0]

num_rotations = 100
for i in range(len(x_train_positives)):
     for r in range(num_rotations):
          rotated_row = np.roll(X_train[i,:], shift = r)
          X_train = np.vstack([X_train, rotated_row])

y_train = np.vstack([y_train, np.array([1] * len(x_train_positives) * num_rotations).reshape(-1,1)])
print(X_train.shape, y_train.shape)

print(pd.Series(y_train.reshape(y_train.shape[0],)).value_counts())
# 0 class: 4040; 1 class: 2929


# Feature Scaling and Reshape
X_train = ((X_train - np.mean(X_train, axis=1).reshape(-1,1)) / np.std(X_train, axis=1).reshape(-1,1))[:,:,np.newaxis]
X_val = ((X_val - np.mean(X_val, axis=1).reshape(-1,1)) / np.std(X_val, axis=1).reshape(-1,1))[:,:,np.newaxis]
X_test = ((X_test - np.mean(X_test, axis=1).reshape(-1,1)) / np.std(X_test, axis=1).reshape(-1,1))[:,:,np.newaxis]


X_train, X_val, X_test = X_train.reshape(len(X_train), -1), X_val.reshape(len(X_val), -1), X_test.reshape(len(X_test), -1)
print(X_train.shape,X_val.shape, y_train.shape, y_val.shape)


# ---------------------------- Model -------------------------------
# create and fit Multilayer Perceptron model
# model = Sequential()
# model.add(Dense(64, input_dim=3197, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy',
             # metrics=['accuracy'])
# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation = 'sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('MLP.h5', monitor='val_loss', mode='max', verbose=1, save_best_only=True)
# fit model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=3, epochs=15, class_weight='auto',
            callbacks=[es, mc])
# load the saved model
saved_model = load_model('MLP.h5')

# history = model.fit(X_train, y_train, batch_size=3, epochs=15, class_weight = 'auto', validation_data=(X_val, y_val),
          # callbacks=[ModelCheckpoint("MLP.hdf5", monitor="val_loss", save_best_only=True)])

# print model structure
model.summary()

# plot loss
N = 15
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("MLP Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Model loss MLP3.png")
plt.close()

# ---------------------------- Prediction and Evaluation -------------------------------
#  evaluate recall and f1 score
y_pred = model.predict(X_test)
threshold = 0.5
print(classification_report(y_test,y_pred >= threshold))
