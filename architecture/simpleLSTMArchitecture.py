import tensorflow as tf
from architecture.architecture import Architecture


class SimpleLSTMArchitecture(Architecture):
	def __init__(self, inputShape, lstmParams):
		with tf.variable_scope("LSTMArchitecture"):
			self._inputShape = inputShape
			self._lstmParams = lstmParams
			super().__init__()

	def generateXOutputs(self, seedInput, length):
		assert length > 0
		with tf.variable_scope("LSTMArchitecture"):

			for i in range(length):
				intermediateOutput = self._network(seedInput[int(-self._lstmParams.fftFrames()+i):], reuse=True)
				seedInput = tf.concat([seedInput, tf.expand_dims(intermediateOutput, 1)], axis=1)

			return seedInput[0]

	def inputShape(self):
		return self._inputShape

	def _lossGraph(self):
		with tf.variable_scope("Loss"):
			targetSquaredNorm = tf.reduce_sum(tf.square(self._target), axis=[1])

			error = self._target - self._output
			error_per_example = tf.reduce_sum(tf.square(error), axis=[1])

			reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (targetSquaredNorm + 1e-4)))
			lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
			total_loss = tf.add_n([reconstruction_loss, lossL2])

			total_loss_summary = tf.summary.scalar("total_loss", total_loss)
			l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
			rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
			self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

			return total_loss

	def _network(self, data, reuse=False):
		with tf.variable_scope("Network", reuse=reuse):
			rnn_cell = tf.contrib.rnn.BasicLSTMCell(self._lstmParams.lstmSize())
			# dataset = tf.split(data, int(self._lstmParams.fftFrames()-1), -2)
			print(data.shape)
			dataset = tf.unstack(data, axis=-2)

			outputs, states = tf.nn.static_rnn(rnn_cell, dataset, dtype=tf.float32)

			# there are n_input outputs but
			# we only want the last output
			output = tf.matmul(outputs[-1], self._weight_variable(
				[self._lstmParams.lstmSize(), self._lstmParams.fftFreqBins()])) + self._bias_variable(
				[self._lstmParams.fftFreqBins()])
			return tf.reshape(output, [-1, self._lstmParams.fftFreqBins()])


	def _weight_variable(self, shape):
		return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

	def _bias_variable(self, shape):
		return tf.get_variable('bias', shape, initializer=tf.contrib.layers.xavier_initializer())
