##--------------------------------Imports-----------------------------------------
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

##----------------------------Data Preprocessing-------------------------------
# Load train and test data
test_data = pd.read_csv('exoTest.csv')
train_data = pd.read_csv('exoTrain.csv')

# Check Shape and Dimension
print(test_data.shape)
print(train_data.shape)

# Take a look at the first few rows
print(test_data.head())
print(train_data.head())

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



# ---------------------------- Model -------------------------------
# Create and fit Multilayer Perceptron model

model = Sequential() # initialize network
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu')) # adding an input layer and the first hidden layer
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu')) # adding the second hidden layer
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu')) # adding the third hidden layer
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid')) # adding the fourth hidden layer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #compiling the MLP
# model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Simple Early Stopping to get the best model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('MLP.h5', monitor='val_loss', mode='max', verbose=1, save_best_only=True)

# Fitting the MLP model to the training set
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=3, epochs=15, class_weight='auto',
            callbacks=[es, mc])
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=50, class_weight='auto',
            # callbacks=[es, mc])

# Load saved model
saved_model = load_model('MLP.h5')

# Print Model Structure
model.summary()

# Plot loss
N = 15
# N = 50
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("MLP Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Model loss MLP.png")
plt.close()

# ---------------------------- Prediction and Evaluation -------------------------------
#  Evaluate f-1 score
y_pred = model.predict(X_test)
threshold = 0.5
print(classification_report(y_test,y_pred >= threshold))
