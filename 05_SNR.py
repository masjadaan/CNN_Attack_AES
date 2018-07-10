# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# * Author : Mahmoud Jadaan                                             *
# *                                                                     *
# * Purpose: Compute the SNR of traces0              *
# ***********************************************************************
import numpy as np
import h5py
np.random.seed(128)

# Variance of signal:
#         1. Sum up many traces(vector of summed traces)
#         2. Divide by the number of those traces (averaged trace)
#         3. compute the variance of the averaged trace


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


def variance_signal(traces):
    """
    Variance of signal for many traces requires two steps:
    1. compute the average of many traces (the averaged trace)
    2. compute the variance of that averaged trace
    The variance of a signal is usually computed as following:
        1. compute the mean (m) of a vector (the averaged trace).
        2. new vector = (element of the averaged trace - m)^2
        3. compute the average (m') of the new vector.
    numpy provides a function that computes the variance.
    :param  float traces: n-dimensional array contains the traces, where the number of rows is the number of traces
                          and columns hold values
    :return array contains variance of the mean vector.
    """
    averaged_trace = compute_averaged_trace(traces)
    variance = np.var(averaged_trace)
    return variance


def get_noise(traces):
    """
    It computes the noise in a trace following the steps:
        1. compute the averaged trace
        2. Noise = raw trace - averaged trace
    :param  float traces: n-dimensional array contains the traces, where the number of rows is the number of traces
                          and columns hold values
    :return n-dimensional array contains the noise.
    """
    averaged_trace = compute_averaged_trace(traces)
    nr_traces = len(traces)
    trace_length = len(traces[0])
    noise = np.zeros(nr_traces * trace_length).reshape(nr_traces, trace_length, 1, 1)  # n-dimensional array
    for i in range(nr_traces):
        noise[i] = np.subtract(traces[i], averaged_trace)  # noise = raw trace - average
    return noise


def variance_noise(traces):
    """
    It computes only the variance of noise in a trace following the steps:
        1. compute the averaged trace
        2. Noise = raw trace - averaged trace
        3. compute the variance of Noise.
    at the end the variances have been averaged
    :param  float traces: n-dimensional array contains the traces, where the number of rows is the number of traces
                          and columns hold values
    :return the averaged variance.
    """
    noise = get_noise(traces)
    nr_traces = len(traces)
    variance = np.zeros(nr_traces)  # vector contains the variances of some traces
    for i in range(nr_traces):
        variance[i] = np.var(noise[i])  # compute the variance of noise
        mean_variance = np.mean(variance)
    return mean_variance


def signal_noise_ratio(traces):
    """
    It computes the signal to noise ratio which is calculated by dividing
    the variance of a signal over the variance of a noise.
    SNR = variance signal/ variance noise
    :param traces: n-dimensional array contains the traces
    :return: the value of snr.
    """
    var_signal = variance_signal(traces)
    print("var_signal", var_signal)
    var_noise = variance_noise(traces)
    print("var_Noise", var_noise)
    snr = var_signal/var_noise
    return snr


# **************************************
# *     read the Compressed traces     *
# **************************************
compTrace_file = h5py.File('Compressed_Traces/Compressed_Traces.hdf5', 'r')
X_train_original = np.array(compTrace_file.get('C_T'))
print("Compressed Traces shape:", X_train_original.shape)
print('**************************\n')


# *****************************************
# * read the Noisy Original traces        *
# *****************************************
Noisy_Original_File = h5py.File('3- CNN Noisy/Noisy_Original_Traces.hdf5', 'r')
Noisy_Original_Traces = np.array(Noisy_Original_File.get('N_O_T'))


# *****************************************
# * read the Noisy 50_shifted traces     *
# *****************************************
Noisy_50_File = h5py.File('4- CNN Noisy Shift 50/Noisy_50_Shifted_Traces.hdf5', 'r')
Noisy_Shifted_50 = np.array(Noisy_50_File.get('N_50_S_T'))


# *****************************************
# * read the Noisy 100_shifted traces     *
# *****************************************
Noisy_100_File = h5py.File('5- CNN Noisy Shift 100/Noisy_100_Shifted_Traces.hdf5', 'r')
Noisy_Shifted_100 = np.array(Noisy_100_File.get('N_100_S_T'))


# *****************************************
# * SNR = Var(signal)/Var(noise)     *
# *****************************************

print('SNR Original_Traces = ', signal_noise_ratio(X_train_original))
print('************************\n')

print('SNR Noisy_Original_Traces = ', signal_noise_ratio(Noisy_Original_Traces))
print('************************\n')

print('SNR Noisy_50_Shifted_Traces = ', signal_noise_ratio(Noisy_Shifted_50))
print('************************\n')

print('SNR Noisy_100_Shifted_Traces = ', signal_noise_ratio(Noisy_Shifted_100))
print('************************\n')


