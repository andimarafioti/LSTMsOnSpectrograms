import tensorflow as tf
import numpy as np


class LSTMPreAndPostProcessor(object):
	def __init__(self, lstmParameters):
		self._lstmParameters = lstmParameters

	def signalLength(self):
		return self._lstmParameters.signalLength()

	def inputAndTarget(self, signal):
		realAndImagStft = self._realAndImagStft(signal)

		netInput = np.empty([64, 4, 65, 2])
		target = np.empty([64, 1, 65, 2])
		frames_width = 4

		for step in range(realAndImagStft.shape.as_list()[-3]-frames_width-1):
			netInput = tf.concat([netInput, realAndImagStft[:, step:step + frames_width, :, :]], -3)
			target = tf.concat([target, realAndImagStft[:, step + frames_width + 1:step + frames_width + 2, :, :]], -3)

		return netInput, target

	def _realAndImagStft(self, signal):
		stft = tf.contrib.signal.stft(signals=signal,
									  frame_length=self._lstmParameters.fftWindowLength(),
									  frame_step=self._lstmParameters.fftHopSize())
		real_part = tf.real(stft)
		imag_part = tf.imag(stft)
		return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
