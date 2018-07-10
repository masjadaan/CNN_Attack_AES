# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# * Purpose: Evaluate the trained model on shifted                      *
# traces by 50 by computing accuracy and mean rank                      *
# ***********************************************************************

import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import time
from keras.models import load_model
np.random.seed(128)


# from https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input
    model_multi_inputs_cond = True

    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps, model):
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    nr_rows = 5
    nr_cols = 5
    for i, activation_map in enumerate(activation_maps):

        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.vstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=1)
        #                 print(activations)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')

        plt.figure(1)
        plt.subplot(nr_rows, nr_cols, i + 1)
        plt.plot(activations)
        # plt.title(model.get_layer(activation_map, i))
    plt.show()


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


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

def key_xor_ptxt(keys, plaintext):
    """
    It xor only the first byte of both the key and the plain text.
    :param float keys     : n-dimensional array contains the keys.
    :param float plaintext: n-dimensional array contains the plain text.
    :return type          : an array holds the results of xoring
    """
    nr_keys = len(keys)
    k_xor_p = np.zeros(nr_keys).astype(int)
    for i in range(nr_keys):
        k_xor_p[i] = np.bitwise_xor(keys[i][0].astype(int), plaintext[i][0].astype(int))
    return k_xor_p


def display_1st_byte_value(key_guesses):

    for i in range(len(key_guesses)):
        k_first_byte = key_guesses[i][0]
        print('first byte of the key of trace %d = ' % (i, k_first_byte))



def compute_score(test_traces, plaintext):
    """
    It receives the traces of the attack phase then computes the score as follow:
        1- First byte if the plaintext XOR all possible number of first byte of the key we get class number
        2- The class number is an index to access the prediction vector
        2- For each trace we sum up the logarithm of prediction[j][class number].
    :param test_traces: n-dimensional array holds the traces from victim device.
    :param plaintext: n-dimensional array contains plaintext
    :return: an array holds the score of each number within the first byte of the key.
    """

    number_traces = len(test_traces)
    length_score = 256
    score_vector = np.zeros(number_traces * length_score).reshape(number_traces, length_score)
    key_1st_byte = np.arange(0, 256)
    number_in_byte = len(key_1st_byte)

    for j in range(number_traces):  # loop through the number of traces. trace[j]
        for k in range(number_in_byte):  # loop through the numbers in first byte.
            class_number = np.bitwise_xor(key_1st_byte[k].astype(int), plaintext[j][0].astype(int))
            # the position of the largest peak is the number of the first byte of a key
            score_vector[j][k] = np.log(prediction[j][class_number])
    return score_vector


def indices_descendant_sorted(ndarray):
    """
    It sorts the indices in descendant order depends on their values. The position of the index with
    the largest peak corresponds to the first key guess, so that those sorted indices are the key
    guesses values
    :param ndarray: It contains the values
    :return: a sorted array where the index with the largest value at the first position
    """
    rows = len(ndarray)
    cols = len(ndarray[0])
    sorted_indices = np.zeros(rows*cols).reshape(rows, cols)
    for i in range(len(ndarray)):
        sorted_indices[i] = np.argsort(ndarray[i])[::-1]
    return sorted_indices.astype(int)


def array_descendant_sorted(ndarray):
    """
    It sorts the values in an array in descendant order.
    :param ndarray: It contains the values to be sorted
    :return: a sorted array where the largest value is at the first position.
    """
    for i in range(len(ndarray)):
        ndarray[i] = np.sort(ndarray[i])[::-1]
    return ndarray


def guess_counter(key_guesses, original_key):
    """
    It receives the key guesses and compare them with the real key values in order to find how many
    guesses are required to find the correct value of the key
    :param key_guesses: It contains the potential key values
    :param original_key: It contain the real value of keys
    :return: an array contain the number of guesses for each key
    """
    number_keys = len(key_guesses)
    number_guesses = len(key_guesses[0])
    match = np.zeros(number_keys).astype(int)

    for i in range(number_keys):
        match_counter = 0
        for j in range(number_guesses):
            if key_guesses[i][j] == original_key[i][0].astype(int):
                match[i] = match_counter
                break
            else:
                match_counter = match_counter + 1
    return match


def plot_signal(x_signal, y_signal, class_number):
    """
    It plots the signal with its class number.
    :param float x_signal : 4 dimensional tensor, its second dimension contains the signal amplitude.
    :param float y_signal : 2 dimensional tensor, its second dimension contains the calss number
    :param int class_number: the desired class number.
    """
    plt.figure(figsize=(5,2))
    sample_length = len(x_signal[0])
    plt.plot(x_signal[class_number].reshape(sample_length,1))
    plt.title("Class {}".format(y_signal.argmax(1)[class_number]))
    plt.show()


# ***********************************
# *     read the Shifted traces     *
# ***********************************
Shifted_Traces_file = h5py.File('shifted_50_Traces.hdf5', 'r')  # read
X_train_original = np.array(Shifted_Traces_file.get('S_50_T'))
print("Shifted Traces shape:", X_train_original.shape)
print('*****************************************************************************************\n')


# **************************************
# *     read the Labels                *
# **************************************
label_file = h5py.File('../256_Original_labels/256_Label.hdf5', 'r')  # read
Y_train_original = np.array(label_file.get('256_L'))
print("The original data:")
print("  X_train_original shape:", X_train_original.shape)
print("  Y_train_original shape:", Y_train_original.shape)


# ********************************
# *     read the Plaintext       *
# ********************************
plaintext_file = h5py.File('../Original_Plaintexts/plaintxt.hdf5', 'r')  # read
Original_Plaintext = np.array(plaintext_file.get('P_txt'))
print("Original Plaintext shape:", Original_Plaintext.shape)


# ***************************
# *     read the Keys       *
# ***************************
key_file = h5py.File('../Original_Keys/keys.hdf5', 'r')  # read
Original_Keys = np.array(key_file.get('O_keys'))
print("Original Keys shape:", Original_Keys.shape)


# **************************************
# *     divide the traces              *
# **************************************
# 1- Test data range:
testDataUpperLimit = data_limit(X_train_original, 0.2)
testDataLowerLimit = 0
x_Test, y_Test = divide_data(X_train_original, Y_train_original, testDataUpperLimit, testDataLowerLimit)
"""
Labels =    27
            205
            255
            119
            26
            8
            147
            154
            164
"""

# **************************************
# *     divide the Plaintext           *
# **************************************
Plaintext = Original_Plaintext[testDataLowerLimit:testDataUpperLimit]


# **************************************
# *     divide the Keys                *
# **************************************
keys = Original_Keys[testDataLowerLimit:testDataUpperLimit]


# *****************************************
# *  Load a trained model and its weights *
# *****************************************
model = load_model("Model_Shifted_50.hdf5")
model.load_weights("Best_weights_Shifted_50.hdf5")


# ***********************************
# *   Evaluate model on test data   *
# ***********************************
batchSize = 200
loss, accuracy = model.evaluate(x_Test, y_Test, batchSize, verbose=0)
print('The Accuracy of Model: {:.2f}%'.format(accuracy * 100))
print('The loss on test data: {:.2f}'.format(loss))


# ***************************************
# *     Prediction of classes           *
# ***************************************

# prediction gives the probability of each class for each trace
# In essence prediction vector tries to be as the label vector as much as it can.
prediction = model.predict(x_Test)


# ***************************************
# *     Plot activation maps            *
# ***************************************
# activation_map = get_activations(model, x_Test[0:1], print_shape_only=True)  # with just one sample.
# display_activations(activation_map, model)


# *******************************************
# *  Compute the score of all possible keys *
# *******************************************

score = compute_score(x_Test, Plaintext)
# key guesses are the positions of the largest peaks in descendant order
key_Guesses = indices_descendant_sorted(score)
# score = array_descendant_sorted(score)
number_guesses_per_key = guess_counter(key_Guesses, keys)

# for i in range(len(number_guesses_per_key)):
#     print('Number of guesses for trace %d = %d' % (i, number_guesses_per_key[i]))

mean_rank = np.mean(number_guesses_per_key)
print('The mean rank = ', mean_rank)


# ***********************************
# *   Close all Open Files          *
# ***********************************
Shifted_Traces_file.close()
label_file.close()

