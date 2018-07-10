# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# *                                                                     *
# * Purpose: runs fifth, it Generates and add noise                     *
# * onto original traces                                                *
# ***********************************************************************
import numpy as np
import h5py
from matplotlib import pyplot as plt
np.random.seed(128)


def generate_noise(nr_samples, sample_length, noise_per_sample, sigma):
    """
    It generates as much noise as required from a normal (Gaussian) distribution.
    :param int nr_samples       : Number of samples that different noise vectors must be added to them.
    :param int sample_length    : Number of elements in the sample vector.
    :param int noise_per_sample : Desired number of noise vectors that each sample should get.
    :param float sigma          : standard deviation, it tells how much the data points are
                        close(for small value) / far (for high value) form the mean
    """
    noise_length = nr_samples * sample_length * noise_per_sample
    new_nr_samples = nr_samples * noise_per_sample
    mu = 0  # mean
    noise = np.random.normal(mu, sigma, noise_length).astype('float32')

    noise = noise.reshape(new_nr_samples, sample_length, 1, 1)
    return noise


def add_noise(new_nr_samples, sample_length, x_train, noise, y_train):
    """
    It add a noise onto a signal.
    :param int new_nr_samples: Desired new number of samples.
    :param int sample_length : Length of a sample.
    :param float x_train     : Original signal without noise.
    :param float noise       : Generated noise.
    :param int y_train       : label of the original signal.
    """
    k = 0
    # creating an array to hold both signals
    pattern_x = np.ndarray((new_nr_samples * sample_length)).reshape(new_nr_samples, sample_length, 1, 1)

    for i in range(int(new_nr_samples / (len(x_train)))):
        for j in range(len(x_train)):
            pattern_x[k] = np.add(x_train[j], noise[k])
            k = k + 1
            # create an array to hold the new labels
    pattern_y = np.copy(y_train)
    for i in range(int(new_nr_samples / (len(x_train))) - 1):
        pattern_y = np.concatenate((pattern_y, y_train))
    return pattern_x, pattern_y


def noisy_key_plaintext(new_nr_samples, key_plaintext, x_train):
    """
    Each trace might get x number of noise traces, which means the total number of traces increases
    so the label array must be concatenated many times equal to the number of noise vectors each trace
    gets, the same applies on the key and plaintext array.
    This function concatenate the key array or plaintext array
    traces = [0, 1, 2, ...n]    [0, 1, 2, ...n].........[0, 1, 2, ...n]
    label = [c0, c1, c2,...n]   [c0, c1, c2,...n].......[c0, c1, c2,...n]
    :param int new_nr_samples: the new number after adding noise
    :param key_plaintext: either the key or plaintext array
    :param x_train: traces before adding noise
    :return: concatenated key or plaintext
    """
    new_key_plaintext = np.copy(key_plaintext)
    for i in range(int(new_nr_samples / (len(x_train))) - 1):
        new_key_plaintext = np.concatenate((new_key_plaintext, key_plaintext))
    return new_key_plaintext


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


# ********************************
# *     read the Plaintext       *
# ********************************
plaintext_file = h5py.File('Original_Plaintexts/plaintxt.hdf5', 'r')  # read
Original_Plaintext = np.array(plaintext_file.get('P_txt'))
print("Original Plaintext shape:", Original_Plaintext.shape)
print('********************************\n')


# ***************************
# *     read the Keys       *
# ***************************
key_file = h5py.File('Original_Keys/keys.hdf5', 'r')  # read
Original_Keys = np.array(key_file.get('O_keys'))
print("Original Keys shape:", Original_Keys.shape)
print('********************************\n')


# **************************************
# *     read the Labels                *
# **************************************
label_file = h5py.File('256_Original_labels/256_Label.hdf5', 'r')
Y_train_original = np.array(label_file.get('256_L'))
print("Y_train_original shape:", Y_train_original.shape)
print('********************************\n')


# **************************************
# *     read the Compressed traces     *
# **************************************
compTrace_file = h5py.File('Compressed_Traces/Compressed_Traces.hdf5', 'r')
X_train_original = np.array(compTrace_file.get('C_T'))
print("Compressed Traces shape:", X_train_original.shape)
print('********************************\n')


# **************************************
# *  Read the 50_shifted traces        *
# **************************************
shifted_50_file = h5py.File("1- CNN Shift 50/shifted_50_Traces.hdf5", "r")
X_train_50_Shifted = np.array(shifted_50_file.get('S_50_T'))
print("X_train_50_Shifted shape:", X_train_50_Shifted.shape)
print('********************************\n')


# **************************************
# *  Read the 100_shifted traces       *
# **************************************
shifted_100_file = h5py.File("2- CNN Shift 100/shifted_100_Traces.hdf5", "r")
X_train_100_Shifted = np.array(shifted_100_file.get('S_100_T'))
print("X_train_100_Shifted shape:", X_train_100_Shifted.shape)
print('********************************\n')


# **************************************
# *  Generating a Noise                *
# **************************************
# each trace gets n different type of noise vectors
Noise_Per_Trace = 1
Nr_Traces = len(X_train_original)
Trace_Length = len(X_train_original[0])
New_Nr_Traces = Noise_Per_Trace * Nr_Traces
sigma = 0.00
Noise = generate_noise(Nr_Traces, Trace_Length, Noise_Per_Trace, sigma)
print("Noise shape:", Noise.shape)
print('********************************\n')


# *******************************************
# *  Adding Noise into Traces Without Shift *
# *******************************************
X_Noisy_Without_Shift, Y_Noisy_Shifted = add_noise(New_Nr_Traces, Trace_Length, X_train_original,
                                                   Noise, Y_train_original)
print("X_Noisy_Without_Shift shape:", X_Noisy_Without_Shift.shape)
print('********************************\n')


# *******************************************
# *  Adding Noise into 50 Shifted Traces    *
# *******************************************
X_Noisy_50_Shift, Y_Noisy_Shifted = add_noise(New_Nr_Traces, Trace_Length, X_train_50_Shifted, Noise, Y_train_original)
print("X_Noisy_50_Shift shape:", X_Noisy_50_Shift.shape)
print('********************************\n')


# *******************************************
# *  Adding Noise into 100 Shifted Traces   *
# *******************************************
X_Noisy_100_Shift, Y_Noisy_Shifted = add_noise(New_Nr_Traces, Trace_Length, X_train_100_Shifted, Noise, Y_train_original)
print("X_Noisy_100_Shift shape:", X_Noisy_100_Shift.shape)
print('********************************\n')


print("Y_Noisy_Shifted shape:", Y_Noisy_Shifted.shape)
print('********************************\n')


# The number of traces increases ==> labels increases ==> key and plaintext must increase to
# calculate the score
# key/plaintext array must be concatenate many time = the number of noise vectors each trace gets

# **************************************
# * Concatenate the Keys               *
# **************************************
Noise_Key = noisy_key_plaintext(New_Nr_Traces, Original_Keys, X_train_original)
print("new key shape", Noise_Key.shape)
print('********************************\n')


# **************************************
# * Concatenate the Plaintext          *
# **************************************
Noise_Plaintext = noisy_key_plaintext(New_Nr_Traces, Original_Plaintext, X_train_original)
print("new plaintext shape", Noise_Plaintext.shape)
print('********************************\n')


# **************************************
# * Store the new keys                 *
# **************************************
Noise_Key_File = h5py.File("3- CNN Noisy/Noise_Keys.hdf5", "w")
set_Noise_Key = Noise_Key_File.create_dataset('N_K', data=Noise_Key)


# **************************************
# * Store the new plaintext            *
# **************************************
Noise_Plaintext_File = h5py.File("3- CNN Noisy/Noise_Plaintext.hdf5", "w")
set_Noise_Plaintext = Noise_Plaintext_File.create_dataset('N_PT', data=Noise_Plaintext)


# **************************************
# * Store the Noisy Label              *
# **************************************
Noisy_Label_File = h5py.File("3- CNN Noisy/Noisy_Labels.hdf5", "w")
set_Noisy = Noisy_Label_File.create_dataset('N_L', data=Y_Noisy_Shifted)


# **************************************
# * Store the Noisy Original traces    *
# **************************************
Noisy_Original_File = h5py.File("3- CNN Noisy/Noisy_Original_Traces.hdf5", "w")
set_Noisy_Original = Noisy_Original_File.create_dataset('N_O_T', data=X_Noisy_Without_Shift)


# **************************************
# * Store the Noisy 50_shifted traces  *
# **************************************
Noisy_50_File = h5py.File("4- CNN Noisy Shift 50/Noisy_50_Shifted_Traces.hdf5", "w")
set_50 = Noisy_50_File.create_dataset('N_50_S_T', data=X_Noisy_50_Shift)

# **************************************
# * Store the Noisy 100_shifted traces *
# **************************************
Noisy_100_File = h5py.File("5- CNN Noisy Shift 100/Noisy_100_Shifted_Traces.hdf5", "w")
set_100 = Noisy_100_File.create_dataset('N_100_S_T', data=X_Noisy_100_Shift)


# # **************************************
# # * Plot samples of shifted traces     *
# # **************************************
Number_Traces = 5
plot_traces(Number_Traces, X_train_50_Shifted, Y_train_original)


# *********************************************
# * Plot samples of Noisy and shifted traces  *
# *********************************************
plot_traces(Number_Traces, X_Noisy_50_Shift, Y_Noisy_Shifted)


# *********************************
# *     Close all open files      *
# *********************************
label_file.close()
Noisy_Label_File.close()
shifted_50_file.close()
Noisy_50_File.close()
shifted_100_file.close()
Noisy_100_File.close()
Noise_Key_File.close()
Noise_Plaintext_File.close()




