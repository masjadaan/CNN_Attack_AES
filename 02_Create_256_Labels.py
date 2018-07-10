# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# *                                                                     *
# * Purpose: It must run third, it Creates 256 classes                  *
# ***********************************************************************


import numpy as np
import h5py
from keras.utils import np_utils
from matplotlib import pyplot as plt


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


def create_256_labels(vector, nr_classes):
    """
    It creates the label of the traces in order to use them in CNN
    :param int vector  : contains the digits resulting from xor, that is actually the number of the set bits
    :param in nr_classes: desired number of classes.
    :return type       : n-dimensional array
    """
    y_label = np_utils.to_categorical(vector, nr_classes)
    return y_label


# ***************************
# *     read the Keys       *
# ***************************
"""
--------------------------------------------------------------------------------------------------
  | keys ||1st| 2nd| 3rd| 4th| 5th|---                                      ---|15ht| 16th|
--------------------------------------------------------------------------------------------------
    [0]   [124. 194.  84. 248.  27. 232. 231. 141. 118.  90.  46.  99.  51. 159. 201. 154.]
    [1]   [171. 178. 205. 198. 155. 180.  84.  17.  14. 130. 116.  65.  33.  61. 220. 135.]
    [2]   [143.  56.  92.  42. 236. 176.  59. 251.  50. 175.  60.  84. 236.  24. 219.  92.]
    [3]   [117. 216. 190.  97. 137. 249.  92. 187. 168. 153.  15. 149. 177. 235. 241. 179.]
    [4]   [ 31.  35.  30. 168.  28. 123. 100. 197.  20. 115.  90. 197.  94.  75. 121.  99.]
    [5]   [ 51. 205. 227.  80.  72.  71.  21.  92. 187. 111.  34.  25. 186. 155. 125. 245.]
    [6]   [152.  50.  56. 224. 121.  77.  61.  52. 188.  95.  78. 119. 250. 203. 108.   5.]
    [7]   [ 54. 148. 179. 175. 226. 240. 228. 158.  79.  50.  21.  73. 253. 130.  78. 169.]
    [8]   [172.  91. 243. 142.  76. 215.  45. 155.   9.  66. 229.   6. 196.  51. 175. 205.]
"""
key_file = h5py.File('Original_Keys/keys.hdf5', 'r')  # read
Original_Keys = np.array(key_file.get('O_keys'))
print("Original Keys shape:", Original_Keys.shape)
print('********************************************************************\n')


# ********************************
# *     read the Plaintext       *
# ********************************
"""
   plaintext = [[103. 198. 105. 115.  81. 255.  74. 236.  41. 205. 186. 171. 242. 251.
         227.  70.]
        [102.  50.  13. 183.  49.  88. 163.  90.  37.  93.   5.  23.  88. 233.
          94. 212.]
        [112. 233.  62. 161.  65. 225. 252. 103.  62.   1. 126. 151. 234. 220.
         107. 150.]
        [  2.  26. 254.  67. 251. 250. 170.  58. 251.  41. 209. 230.   5.  60.
         124. 148.]
        [  5. 239. 247.   0. 233. 161.  58. 229. 202.  11. 203. 208.  72.  71.
         100. 189.]
        [ 59. 112. 100.  36.  17. 158.   9. 220. 170. 212. 172. 242.  27.  16.
         175.  59.]
        [ 11. 225.  26.  28. 127.  35. 248.  41. 248. 164.  27.  19. 181. 202.
          78. 232.]
        [172. 134.  33.  43. 170.  26.  85. 162. 190. 112. 181. 115.  59.   4.
          92. 211.]
        [  8. 112. 212. 178. 138.  41.  84.  72. 154.  10. 188. 213.  14.  24.
         168.  68.]]
"""
plaintext_file = h5py.File('Original_Plaintexts/plaintxt.hdf5', 'r')  # read
Original_Plaintext = np.array(plaintext_file.get('P_txt'))
print("Original Plaintext shape:", Original_Plaintext.shape)
print('********************************************************************\n')


# ******************************
# *     key XOR plaintext      *
# ******************************
"""
    k[i] XOR p[i] = 27
                    205
                    255
                    119
                    26
                    8
                    147
                    154
                    164
"""
k_XOR_p = key_xor_ptxt(Original_Keys, Original_Plaintext)
print('**********************************************************************\n')

# ******************************
# *     create 256 labels      *
# ******************************
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
NrClasses = 256
Y_label = create_256_labels(k_XOR_p, NrClasses)
print("label shape", Y_label.shape)
label_file = h5py.File("256_Original_labels/256_Label.hdf5", "w")
labelSet = label_file.create_dataset('256_L', data=Y_label)
print('256 classes have been created\n')


# *********************************
# *     Close all open files      *
# *********************************
plaintext_file.close()
key_file.close()
label_file.close()

