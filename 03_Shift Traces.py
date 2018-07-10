# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# *                                                                     *
# * Purpose: runs fourth in case a shift is required                    *
# * it Introduces random shift onto original traces                     *
# ***********************************************************************
import numpy as np
import h5py
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(128)


def compute_averaged_trace(traces):
    """
    It computes the averaged trace(not average of a trace) of many traces following the steps:
        1. Sum up many traces(vector of summed traces)
        2. Divide by the number of those traces (averaged trace)
    :param  float traces: n-dimensional array contains the traces, where the number of rows is the number of traces
                          and columns hold values
    :return vector contains averaged trace
    """
    nr_traces = len(traces)
    averaged_trace = traces[0]
    for i in range(1, nr_traces):  # starts at 1 cuz trace[0] is already assigned
        averaged_trace = np.add(averaged_trace, traces[i])
    averaged_trace = np.divide(averaged_trace, nr_traces)
    return averaged_trace


def sort_HW(traces, setbits):
    """
    It receives the traces and a vector that contains the number of set bits after XORing the
    Keys and Plain texts. [trace i] has [n of set bits]
    :param traces : n-dimensional array. traces[Number of traces][length][1][1]
    :param setbits: Holds the number of set bits.
    :return: a list holds traces sorted into 9 different classes.
    """
    HWclasses = [[], [], [], [], [], [], [], [], []]
    NrTraces = len(setbits)
    for i in range(NrTraces):
        if (setbits[i] == 0):
            HWclasses[0].append(traces[i])
        elif (setbits[i] == 1):
            HWclasses[1].append(traces[i])
        elif (setbits[i] == 2):
            HWclasses[2].append(traces[i])
        elif (setbits[i] == 3):
            HWclasses[3].append(traces[i])
        elif (setbits[i] == 4):
            HWclasses[4].append(traces[i])
        elif (setbits[i] == 5):
            HWclasses[5].append(traces[i])
        elif (setbits[i] == 6):
            HWclasses[6].append(traces[i])
        elif (setbits[i] == 7):
            HWclasses[7].append(traces[i])
        elif (setbits[i] == 8):
            HWclasses[8].append(traces[i])
    return HWclasses


def data_aug(x_train, y_train, shift_value):
    """
    It uses Keras data augmentation function to generate randomly shifted data with its labels
    :param float x_train: 4 dimensional tensor, its second dimension contains amplitude.
    :param float y_train: 2 dimensional tensor, its second dimension contains the calss number.
    :param float shift_value: the desired shift.
    """
    datagen = ImageDataGenerator(height_shift_range=shift_value)

    i = 0
    for x_train_aug, y_train_aug in datagen.flow(x_train, y_train, batch_size=len(x_train), shuffle=False):
        """
        The .flow() generates batches of randomly transformed images and saves the results.
        :param int batch_size : indicate how many samples from X_train we want to use.
        :param boolean shuffle: To shuffle the data each round.
        """
        i += 1
        if i > 10:
            break  # otherwise the generator would loop indefinitely
    return x_train_aug, y_train_aug


def plot_traces(nr_traces, x_traces, y_traces):
    """
    It plot specified number of traces in one figure
    :param int nr_traces: desired number of traces
    :param float x_traces: n-dimensional array holds the traces
    :param y_traces: n-dimensional array holds labels
    """
    t = np.arange(0, len(x_traces[0]))
    for i in range(nr_traces):
        plt.figure(1)
        plt.subplot(nr_traces, 1, i + 1)
        plt.plot(t, x_traces[i].reshape(len(x_traces[0]), 1))
        plt.title("Class {}".format(y_traces.argmax(1)[i]))  # print the labels
    plt.show()


# **************************************
# *     read the Compressed traces     *
# **************************************
compressed_file = h5py.File('Compressed_Traces/Compressed_Traces.hdf5', 'r')
Compressed_Traces = np.array(compressed_file.get('C_T'))
print("Compressed Traces shape:", Compressed_Traces.shape)


# **************************************
# *     read the Labels                *
# **************************************
label_file = h5py.File('256_Original_labels/256_Label.hdf5', 'r')
Y_train_original = np.array(label_file.get('256_L'))
print("Y_train_Shifted shape:", Y_train_original.shape)


# *******************************************
# *     Shift the Compressed traces by 50   *
# *******************************************
Shift_50 = 0.1  # random shift range = 0 to 50
X_train_Shifted_50, Y_train_Shifted = data_aug(Compressed_Traces, Y_train_original, Shift_50)


# *******************************************
# *    Shift the Compressed traces by 100   *
# *******************************************
Shift_100 = 0.2  # random shift range = 0 to 100
X_train_Shifted_100, Y_train_Shifted = data_aug(Compressed_Traces, Y_train_original, Shift_100)


# **************************************
# *  Store the 50_shifted traces       *
# **************************************
shifted_50_file = h5py.File("1- CNN Shift 50/shifted_50_Traces.hdf5", "w")
compSet_50 = shifted_50_file.create_dataset('S_50_T', data=X_train_Shifted_50)


# **************************************
# *  Store the 100_shifted traces      *
# **************************************
shifted_100_file = h5py.File("2- CNN Shift 100/shifted_100_Traces.hdf5", "w")
compSet_100 = shifted_100_file.create_dataset('S_100_T', data=X_train_Shifted_100)


# # ***********************************************
# # *     locate the information in a trace       *
# # ***********************************************
# #compute the averaged trace of all traces
# aveTraces = AveragedTrace(Original_Traces)
#
# # compute the average trace of each classes
# classesAveTrace = np.zeros(NrClasses*len(Original_Traces[0])).reshape(NrClasses, len(Original_Traces[0]), 1, 1)
#
# for i in range(NrClasses):
#     classesAveTrace[i] = AveragedTrace(np.array(HWclasses[i]))
#
# #compute the difference between the average trace of each classes and the averaged trace of all traces(information)
# classesInfo = np.zeros(NrClasses*len(Original_Traces[0])).reshape(NrClasses, len(Original_Traces[0]), 1, 1)
# for i in range(NrClasses):
#     classesInfo[i] = np.subtract(classesAveTrace[i], aveTraces)
# #     plotSignal(classesInfo, i)


# **************************************
# *  Plot samples of original traces   *
# **************************************
Number_Traces = 8
plot_traces(Number_Traces, Compressed_Traces, Y_train_original)


# **************************************
# * Plot samples of 50 shifted traces *
# **************************************
plot_traces(Number_Traces, X_train_Shifted_50, Y_train_original)

# **************************************
# * Plot samples of 100 shifted traces *
# **************************************
plot_traces(Number_Traces, X_train_Shifted_100, Y_train_original)


# *********************************
# *     Close all open files      *
# *********************************
label_file.close()
compressed_file.close()
shifted_50_file.close()
shifted_100_file.close()
