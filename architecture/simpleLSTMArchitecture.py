import tensorflow as tf
import numpy as np
from architecture.architecture import Architecture


class SimpleLSTMArchitecture(Architecture):
	def __init__(self, inputShape, lstmParams):
		with tf.variable_scope("LSTMArchitecture"):
			self._inputShape = inputShape
			self._lstmParams = lstmParams
			self._state = None
			super().__init__()

	def generateXOutputs(self, seedInput, length):
		assert length > 0
		with tf.variable_scope("LSTMArchitecture"):
			self._state = None

			intermediateOutput = self._network(seedInput[:, int(-self._lstmParams.fftFrames()):, :], reuse=True)
			seedInput = tf.concat([seedInput, intermediateOutput[:, -1:, :]], axis=1)
			print(seedInput.shape.as_list()[1])

			for i in range(seedInput.shape.as_list()[1], length):
				intermediateOutput = self._network(seedInput[:, -1:, :], initial_state=self._state, reuse=True)
				seedInput = tf.concat([seedInput, intermediateOutput[:, -1:, :]], axis=1)

			return seedInput

	def inputShape(self):
		return self._inputShape

	def _lossGraph(self):
		with tf.variable_scope("Loss"):
			# normalize values !! divide by max input and multiply output

			targetSquaredNorm = tf.reduce_sum(tf.square(self._target), axis=[1, 2])

			error = self._target - self._output
			error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2])

			reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (targetSquaredNorm + 1e-4)))
			lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
			total_loss = tf.add_n([reconstruction_loss, lossL2])

			total_loss_summary = tf.summary.scalar("total_loss", total_loss)
			l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
			rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
			self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

			return total_loss

	# def _network(self, data, initial_state=None, reuse=False):
	# 	with tf.variable_scope("Network", reuse=reuse):
	# 		rnn_cell = tf.contrib.rnn.BasicLSTMCell(self._lstmParams.lstmSize())
	# 		# dataset = tf.split(data, int(self._lstmParams.fftFrames()-1), -2)
	# 		dataset = tf.unstack(data, axis=-2)
    #
	# 		outputs, self._state = tf.nn.static_rnn(rnn_cell, dataset, initial_state=initial_state, dtype=tf.float32)
    #
	# 		out_output = np.empty([data.shape[0], 0, self._lstmParams.fftFreqBins()])
	# 		weights = self._weight_variable([self._lstmParams.lstmSize(), self._lstmParams.fftFreqBins()])
	# 		biases = self._bias_variable([self._lstmParams.fftFreqBins()])
    #
	# 		for output in outputs:
	# 			mat_muled = tf.matmul(output, weights) + biases
	# 			output = tf.reshape(mat_muled, [-1, 1, self._lstmParams.fftFreqBins()])
	# 			out_output = tf.concat([out_output, output], axis=1)
	# 		return out_output

	def _lstmNetwork(self, data, initial_state, reuse, name):
		with tf.variable_scope(name, reuse=reuse):
			dataset = tf.unstack(data, axis=-2)

			rnn_cell = tf.contrib.rnn.MultiRNNCell(
				[tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize()),
				 tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize()),
				 tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize())])
			outputs, states = tf.nn.static_rnn(rnn_cell, dataset, initial_state=initial_state, dtype=tf.float32)

			out_output = np.empty([data.shape[0], 0, self._lstmParams.fftFreqBins()])
			weights = self._weight_variable([self._lstmParams.lstmSize(), self._lstmParams.fftFreqBins()])
			biases = self._bias_variable([self._lstmParams.fftFreqBins()])

			for output in outputs:
				mat_muled = tf.matmul(output, weights) + biases
				output = tf.expand_dims(mat_muled, axis=1)
				out_output = tf.concat([out_output, output], axis=1)
			return out_output, states

	def _network(self, data, reuse=False, initial_state=None):
		with tf.variable_scope("Network", reuse=reuse):
			# initialize LSTM states with data
			forward_lstmed, self._state = self._lstmNetwork(data, initial_state, reuse, 'forward_lstm')
			output_audio = forward_lstmed[:, -4:, :]

			for i in range(int(self._lstmParams.outputWindowCount())+3):
				next_frame, forward_states = self._lstmNetwork(output_audio[:, -1:, :], self._state, True,
															   'forward_lstm')
				output_audio = tf.concat([output_audio, next_frame], axis=1)

			predictedFrames = self._predictNetwork(output_audio, reuse)
			return predictedFrames

	def _predictNetwork(self, mixed_gaps, reuse):
		with tf.variable_scope("predict", reuse=reuse):
			mixing_variables = self._weight_variable(
				[self._lstmParams.fftFreqBins() * 7, self._lstmParams.fftFreqBins()])
			output = tf.zeros([self._lstmParams.batchSize(), 0, self._lstmParams.fftFreqBins()])
			for i in range(3, self._lstmParams.outputWindowCount()+3):
				intermediate_output = tf.reshape(mixed_gaps[:, i-3:i+4], (self._lstmParams.batchSize(),
																		  self._lstmParams.fftFreqBins() * 7))
				intermediate_output = tf.matmul(intermediate_output, mixing_variables)
				intermediate_output = tf.expand_dims(intermediate_output, axis=1)
				output = tf.concat([output, intermediate_output], axis=1)
			return output

	def _weight_variable(self, shape):
		return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

	def _bias_variable(self, shape):
		return tf.get_variable('bias', shape, initializer=tf.contrib.layers.xavier_initializer())
