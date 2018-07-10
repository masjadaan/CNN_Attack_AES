# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# *                                                                     *
# * Purpose: It must run second to Compress the length of               *
# * original traces from 24999 to 500                                   *
# ***********************************************************************


import numpy as np
import h5py
from matplotlib import pyplot as plt


def compress_traces(traces):
    """
    It receives traces and compress them by taking a sample every 50 step creating
    a compressed traces of length 500.
    traces[Number of traces][length][1][1]
    :param traces: n-dimensional array. traces[Number of traces][length][1][1]
    :return: compressed traces with their new length
    """
    nr_traces = len(traces)
    trace_length = len(traces[0])
    compress_factor = 50
    com_trace_length = int((len(traces[0]) + 1) / compress_factor)  # 25000/50 = 500
    compressed_traces = np.zeros(nr_traces * com_trace_length).reshape(nr_traces, com_trace_length, 1, 1)
    for i in range(nr_traces):  # loop through number of traces
        k = 0
        for j in range(trace_length):  # loop through a trace's length
            if (j % compress_factor) == 0:
                compressed_traces[i][k] = traces[i][j]
                k = k + 1
    return compressed_traces


def plot_traces(nr_traces, x_traces):
    """
    It plot specified number of traces in one figure
    :param int nr_traces: desired number of traces
    :param float x_traces: n-dimensional array holds the traces
    """
    t = np.arange(0, len(x_traces[0]))
    for i in range(nr_traces):
        plt.figure(1)
        plt.subplot(nr_traces, 1, i + 1)
        plt.plot(t, x_traces[i].reshape(len(x_traces[0]), 1))
    plt.show()


# ********************************
# *     read the traces          *
# ********************************
file1 = h5py.File('Original_Traces/Original_Traces.hdf5', 'r')
Original_Traces = np.array(file1.get('O_T'))
print("Original_Traces shape:", Original_Traces.shape)
print('*******************************************************\n')


# ***********************************
# *  compress and store the traces  *
# ***********************************
Compressed_Traces = compress_traces(Original_Traces)
print('The compresses traces shape', Compressed_Traces.shape)
compf = h5py.File("Compressed_Traces/Compressed_Traces.hdf5", "w")
compSet = compf.create_dataset('C_T', data=Compressed_Traces)


# **************************************
# * Plot samples of Compressed traces  *
# **************************************
Number_Traces = 8
plot_traces(Number_Traces, Compressed_Traces)


# *********************************
# *     Close all open files      *
# *********************************
file1.close()
compf.close()
