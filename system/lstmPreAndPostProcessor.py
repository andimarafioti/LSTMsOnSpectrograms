import tensorflow as tf
import numpy as np


class LSTMPreAndPostProcessor(object):
	def __init__(self, lstmParameters):
		self._lstmParameters = lstmParameters

	def inputAndTarget(self, signal):
		realAndImagStft = self._realAndImagStft(signal)

		netInput = realAndImagStft[:, :-1, :]
		target = realAndImagStft[:, -1, :]

		return netInput, target

		# netInput = np.empty([0, self._lstmParameters.fftFreqBins()])
		# target = np.empty([0, self._lstmParameters.fftFreqBins()])
		# frames_width = self._lstmParameters.countOfFrames()
		#
		# for step in range(realAndImagStft.shape.as_list()[-2]-frames_width-1):
		# 	netInput = tf.concat([netInput, realAndImagStft[0, step:step + frames_width, :]], -2)
		# 	target = tf.concat([target, realAndImagStft[0, step + frames_width + 1:step + frames_width + 2, :]], -2)
		#
		# return netInput, target

	def _realAndImagStft(self, signal):
		stft = tf.contrib.signal.stft(signals=signal,
									  frame_length=self._lstmParameters.fftWindowLength(),
									  frame_step=self._lstmParameters.fftHopSize())
		return tf.abs(stft)
		# real_part = tf.real(stft)
		# imag_part = tf.imag(stft)
		# return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
