Attacking protected AES with Convolutional Neural Networks
==========================================================

Convolutional Neural Network
----------------------------

Supervised Learning (Labeled Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each image has one class among three classes (Horse, Dog, Cat)

.. image:: /Photos/SVL/horse.png
   :scale: 50

Each trace has one class among two classes (Bit = 0, Bit = 1)

.. image:: /Photos/SVL/pt.png

Network Architecture
~~~~~~~~~~~~~~~~~~~~

.. image:: /Photos/CNN_Layers/5.png

Why CNNs in SCA?
~~~~~~~~~~~~~~~~

	* Can handle high dimensional data.
	* Robust against trace misalignments.
	* Can deal with masking.
	* Find Point of Interests (POIs).

How to measure CNN’s Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	* Accuracy %: Correct class prediction.
.. image:: /Photos/Accu_And_Mean/accuracy.jpg

	* Rank Function: Correct key prediction.
.. image:: /Photos/Accu_And_Mean/Rankings.jpeg


Datasets
--------

Canright AES on FPGA
~~~~~~~~~~~~~~~~~~~~

Original Traces:

.. image:: /Photos/Dataset/3_original_traces.png

.. image:: /Photos/Dataset/Org_T.png


Labeled Traces:

.. image:: /Photos/Dataset/4_Labelled_compressed_traces.png

.. image:: /Photos/Dataset/Comp_T.png


Attacking The Original Traces
------------------------------

Attacking Random shifted Traces
--------------------------------

Attacking Noisy Traces
-----------------------

Increasing The Number Of Traces
-------------------------------

Applying Data Augmentation Technique
------------------------------------


Conclusion
==========

Advantages
----------

	* The realignment phase and the selection of points of interest problem have been removed.
	* Using Data Augmentation in a correct way might improve the result.
	* The tanh loss function performed the best for this dataset.
	* Increasing the number of layers better than increasing the number of filters due to time factor.

Disadvantages
-------------

	* CNNs perform poorly on a traces with low SNR.
	* Each target requires different tuning.
	* Learning time is linear that means the more complex a model is, the more time the learning process needs.
































