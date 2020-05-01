import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM, concatenate, Activation, MaxPool1D, Flatten
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


##----------------------------Data preprocessing-------------------------------
# Load train and test data
test_data = pd.read_csv('exoTest.csv')
train_data = pd.read_csv('exoTrain.csv')

# encoding the label (1 to 0, 2 to 1)
train_data['LABEL'] = train_data['LABEL'].replace([1], [0])
train_data['LABEL'] = train_data['LABEL'].replace([2], [1])
test_data['LABEL'] = test_data['LABEL'].replace([1], [0])
test_data['LABEL'] = test_data['LABEL'].replace([2], [1])

# split X, y, X_test, y_test
X = train_data.iloc[:, 1:]
y = train_data.iloc[:,:1]
# print(y.shape)
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:,:1]

# plot the features
# for i in range(10):
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
# for j in range(10):
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

X = reduce_upper_outliers(X)
X_test = reduce_upper_outliers(X_test)


X = X.values
X_test = X_test.values


#train, validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3,  stratify=y, random_state = 123)

## data scaling and reshape
X_train = ((X_train - np.mean(X_train, axis=1).reshape(-1,1)) / np.std(X_train, axis=1).reshape(-1,1))[:,:,np.newaxis]
X_val = ((X_val - np.mean(X_val, axis=1).reshape(-1,1)) / np.std(X_val, axis=1).reshape(-1,1))[:,:,np.newaxis]
X_test = ((X_test - np.mean(X_test, axis=1).reshape(-1,1)) / np.std(X_test, axis=1).reshape(-1,1))[:,:,np.newaxis]

# print(X_train.shape, X_val.shape, X_test.shape)
# print(y_train.shape, y_val.shape, type(X_train),type(y_train))

# create batch generator
def batch_generator(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    #convert y_train to array
    y_train = np.array(y_train.values)

    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

    yes_idx = np.where(y_train[:, 0] == 1.)[0]
    non_idx = np.where(y_train[:, 0] == 0.)[0]

    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)

        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis=0)

        yield x_batch, y_batch


batch_generator(X_train, y_train, 32)


# RNN model
##------------------------------ RNN Model ---------------------------------

ip = Input(shape=(3197,1))
x = Permute((2, 1))(ip)
x = GRU(16, return_sequences=True)(x)
x = GRU(32, return_sequences=True)(x)
x = GRU(64, return_sequences=True)(x)
x = GRU(128)(x)
x = Dropout(0.25)(x)

y = Conv1D(filters=16, kernel_size=11, activation='relu')(ip)
y = MaxPool1D(strides=4)(y)
y = BatchNormalization()(y)
y = Conv1D(filters=32, kernel_size=11, activation='relu')(y)
y = MaxPool1D(strides=4)(y)
y = BatchNormalization()(y)
y = Conv1D(filters=64, kernel_size=11, activation='relu')(y)
y = MaxPool1D(strides=4)(y)
y = BatchNormalization()(y)
y = Conv1D(filters=128, kernel_size=11, activation='relu')(y)
y = MaxPool1D(strides=4)(y)
y = Flatten()(y)
y = Dropout(0.25)(y)
y = Dense(64, activation='relu')(y)

added = concatenate([x, y])
added = Dense(32, activation='relu')(added)
out = Dense(1, activation='sigmoid')(added)

model = Model(ip, out)

model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

best_weights_filepath = './RNN(best)_model.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)

saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', save_best_only=True, mode='auto')


History = model.fit_generator(batch_generator(X_train, y_train, 32),
                           validation_data=(X_val, y_val),
                           epochs=50,
                           steps_per_epoch=X_train.shape[1]//32, verbose=0,
                           class_weight='auto',
                           callbacks=[earlyStopping, saveBestModel])



plt.style.use("ggplot")
plt.figure()
plt.plot(History.history["loss"], label="train_loss")
plt.plot(History.history["val_loss"], label="val_loss")

plt.title("RNN Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Model loss RNN.png")
plt.close()
##------------------------------ Prediction ---------------------------------
y_pred = model.predict(X_test)
threshold = 0.5
print(classification_report(y_test,y_pred >= threshold))
