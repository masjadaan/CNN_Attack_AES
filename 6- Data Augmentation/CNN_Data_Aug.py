# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# * Purpose: Train CNN model                                            *
# ***********************************************************************


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
np.random.seed(128)


def get_weights_conv1(model, conv1_filters, filter_width, filter_height):
    """
    It plots the weights of the filters in the first convolutional layer.
    :param model: Model of the CNN
    :param int conv1_filters: Number of filters in first convolution layer.
    :param int filter_width : Width of the filter.
    :param int filter_height: Height of the filter.
    """
    weights = model.layers[0].get_weights()
    weights = weights[0].reshape(1, filter_width, filter_height, conv1_filters)
    #     print("The weights of the filter\n",weights)
    print("Filters of Conv1: height, width, #filters:", weights[0].shape)
    weights = np.transpose(weights[0])
    for i in range(len(weights)):
        plt.plot(weights[i])
        plt.title("Filter {}".format(i + 1))  # print the labels
        plt.show()


def divide_data(x_data, y_data, upper_limit, lower_limit):
    """
    It divides the data set into sections
    :param float x_data  : n-dimensional array contains the data.
    :param int y_data    : Labels of the data.
    :param int upper_limit: Data upper limit.
    :param int lower_limit: Data lower limit.
    :return: Divided data and its labels.
    """
    x_train = x_data[lower_limit: upper_limit]
    y_train = y_data[lower_limit: upper_limit]
    return x_train, y_train


def data_limit(x_train, ratio):
    """
    It limits the dataset, usually 20% for test and validation
    :param float x_train: n-dimensional array contains the data.
    :param float ratio  : ratio of the original data.
    """
    x = int(len(x_train) * ratio)
    return x


class TimeHistory(keras.callbacks.Callback):
    """
    It returns the a list contains the time for each epoch per second
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def ArchModel(input_width, input_height, filter_width, filter_height, depth, number_classes):
    """
    It defines the architecture of the model, (C --> P)*2 --> FC.
    Note: the number of classes is defined as 10 classes, but can be modified as required
    :param int input_width  : Width of the inputed data.
    :param int input_height : Height of the inputed data.
    :param int filter_width : Width of the filter.
    :param int filter_height: Height of the filter.
    :param int depth       : Depth of the input.
    :param int number_classes   : desired number of classes.
    """
    model = Sequential()
    dropout_factor = 0.5
    # activation relu , tanh
    Act = "tanh"
    # ----------------------------------------------------------------------------------------------------------------------

    block_1_filters = 8
    # Conv
    model.add(Conv2D(block_1_filters, kernel_size=(filter_height, filter_width), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True,
                     input_shape=(input_height, input_width, depth)))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    block_2_filters = 16
    # Conv
    model.add(Conv2D(block_2_filters, kernel_size=(filter_height, filter_width), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    block_3_filters = 32
    # Conv
    model.add(Conv2D(block_3_filters, kernel_size=(filter_height, filter_width), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    block_4_filters = 64
    # Conv
    model.add(Conv2D(block_4_filters, kernel_size=(filterHeight, filterWidth), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    block_5_filters = 128
    # Conv
    model.add(Conv2D(block_5_filters, kernel_size=(filterHeight, filterWidth), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    block_6_filters = 128
    # Conv
    model.add(Conv2D(block_6_filters, kernel_size=(filterHeight, filterWidth), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    block_7_filters = 128
    # Conv
    model.add(Conv2D(block_7_filters, kernel_size=(filterHeight, filterWidth), padding='same',
                     data_format="channels_last", activation=Act, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------

    # FC
    # The dropping out factor
    final_dense_output = number_classes
    model.add(Flatten())
    model.add(Dense(1024, activation=Act))
    # model.add(Dropout(dropout_factor))
    model.add(Dense(final_dense_output, activation='softmax'))
    return model


# ******************************************************
# * read the augmented traces (in CNN Noisy folder)    *
# ******************************************************
Noisy_File = h5py.File('../3- CNN Noisy/Noisy_Original_Traces.hdf5', 'r')
X_train_original = np.array(Noisy_File.get('N_O_T'))


# **************************************
# *     read the augmented Labels      *
# **************************************
label_file = h5py.File('../3- CNN Noisy/Noisy_Labels.hdf5', 'r')  # read
Y_train_original = np.array(label_file.get('N_L'))
print("The original data:")
print("  X_train_original shape:", X_train_original.shape)
print("  Y_train_original shape:", Y_train_original.shape)


# **************************************
# *     divide the traces              *
# **************************************
# 1- Test data range:
testDataUpperLimit = data_limit(X_train_original, 0.2)
testDataLowerLimit = 0
x_Test, y_Test = divide_data(X_train_original, Y_train_original, testDataUpperLimit, testDataLowerLimit)

# 2- Validation data range:
valDataUpperLimit = 2 * data_limit(X_train_original, 0.2)
valDataLowerLimit = testDataUpperLimit
x_Validation, y_Validation = divide_data(X_train_original, Y_train_original, valDataUpperLimit, valDataLowerLimit)

# 3- training data range:
trainDataUpperLimit = data_limit(X_train_original, 1)
trainDataLowerLimit = valDataUpperLimit
x_Train, y_Train = divide_data(X_train_original, Y_train_original, trainDataUpperLimit, trainDataLowerLimit)

print("-----------------------------------------------------------")
print("The divided data:")
print("  x_Train      :", x_Train.shape)
print("  x_Validation :", x_Validation.shape)
print("  x_Test       :", x_Test.shape)

print("-----------------------------------------------------------")
print("convert the output into classes:")
print("  y_Train      :", y_Train.shape)
print("  y_Validation :", y_Validation.shape)
print("  y_Test       :", y_Test.shape)


# ************************************
# *     Input Dimensions             *
# ************************************
InputWidth = 1
InputHeight = len(x_Train[0])
depth = 1


# ************************************
# *     Filter Dimensions             *
# ************************************
filterWidth = 1
filterHeight = 7

NrClasses = 256
model = ArchModel(InputWidth, InputHeight, filterWidth, filterHeight, depth, NrClasses)

# summarize the architecture
model.summary()


# ***********************************
# *     Compile model               *
# ***********************************
Adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# *********************************
# *     Train model               *
# *********************************
time_callback = TimeHistory()
# checkpoint
# 1- store the best weights
file_path = "Best_weights_Augmented.hdf5"
best_Accuracy_CheckPoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# 2- Early Stop
Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
callbacks_list = [best_Accuracy_CheckPoint, time_callback, Early_Stop]

# Training factors:
batchSize = 200
epochs = 50
model.fit(x_Train, y_Train, batchSize, epochs, verbose=1, validation_data=(x_Validation, y_Validation),
          shuffle=1, callbacks=callbacks_list)

timePerEpoch = time_callback.times
# sum up the time of all epochs , divide by 60 to convert to min
t = np.sum(timePerEpoch)/60
print(t)


# ***********************************
# *   Save the model                *
# ***********************************
model.save("Model_Augmented.hdf5")


# ***********************************
# *   Close all Open Files           *
# ***********************************
Noisy_File.close()
label_file.close()
