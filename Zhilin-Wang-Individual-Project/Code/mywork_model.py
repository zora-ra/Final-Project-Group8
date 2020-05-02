##----------------------------  Load packages  -------------------------------

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Pdf")
import matplotlib.pyplot as plt
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten,AvgPool1D, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model as lm
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import classification_report
##----------------------------Data preprocessing-------------------------------

# Load train and test data
test_data = pd.read_csv('exoTest.csv')
train_data = pd.read_csv('exoTrain.csv')

# Check Shape and Dimension
print('test data shape')
print(test_data.shape)
print('train data shape')
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
X = train_data.iloc[:3500, 1:]
y = train_data.iloc[:3500,:1]

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
        for j in range(remove):
            idx = sorted_values.index[j]
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            for k in range(2 * half_width + 1):
                idx2 = idx_num + k - half_width
                if idx2 < 1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.' + str(idx2)]  # corrected from 'FLUX-' to 'FLUX.'

                count += 1
            new_val /= count  # count will always be positive here
            if new_val < values[idx]:  # just in case there's a few persistently high adjacent values
                df.iloc[i][idx] = new_val
    return df
# reference: https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration

X = reduce_upper_outliers(X).values
X_test = reduce_upper_outliers(X_test).values

#train, validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

# generate more data using rotation
x_train_positives = X_train[np.squeeze(y_train) == 1]
x_train_negatives = X_train[np.squeeze(y_train) == 0]

num_rotations = 100

print('Number of exoplanet:')
print(len(x_train_positives))

print(X_train.shape)
print(type(X_train))

#rotating data to get more exoplanets
for i in range(len(x_train_positives)):
    for r in range(num_rotations):
        rotated_row = np.roll(x_train_positives[i,:],r)
        X_train = np.vstack([X_train, rotated_row])
print(X_train.shape)

y_train = np.vstack([y_train, np.array([1] * len(x_train_positives) * num_rotations).reshape(-1,1)])

## data scaling and reshape
X_train = ((X_train - np.mean(X_train, axis=1).reshape(-1,1)) / np.std(X_train, axis=1).reshape(-1,1))[:,:,np.newaxis]
X_val = ((X_val - np.mean(X_val, axis=1).reshape(-1,1)) / np.std(X_val, axis=1).reshape(-1,1))[:,:,np.newaxis]
X_test = ((X_test - np.mean(X_test, axis=1).reshape(-1,1)) / np.std(X_test, axis=1).reshape(-1,1))[:,:,np.newaxis]
# ##--------------------------------------------- Model ----------------------------------------------------------------
#build the model
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#model compile and train
model.compile(optimizer='SGD', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=512, epochs=50, class_weight = 'auto', validation_data=(X_val, y_val),
          callbacks=([EarlyStopping(monitor = 'val_loss',mode = 'min',verbose = 1, patience = 20),
                      ModelCheckpoint("CNN_model.hdf5", monitor="val_loss", save_best_only =True)]))

# plot loss
model = lm('CNN_model.hdf5')
plt.style.use('ggplot')
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig("Model loss.png")

#---------------------------- Prediction and Evaluation -------------------------------
#get the predict values
y_pred = model.predict(X_test)
threshold = 0.5
print(classification_report(y_test,y_pred >= threshold))
