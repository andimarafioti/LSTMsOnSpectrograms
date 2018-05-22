import tensorflow as tf
import numpy as np
from architecture.architecture import Architecture


class SimpleLSTMArchitecture(Architecture):
	def __init__(self, inputShape, lstmParams):
		with tf.variable_scope("LSTMArchitecture"):
			self._inputShape = inputShape
			self._lstmParams = lstmParams
			super().__init__()

	def inputShape(self):
		return self._inputShape

	def _lossGraph(self):
		with tf.variable_scope("Loss"):
			targetSquaredNorm = tf.reduce_sum(tf.square(self._target), axis=[1, 2])

			error = self._target - self._output
			error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2])

			reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (targetSquaredNorm + 1e-4)))
			lossL2 = 0.0#tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
			total_loss = tf.add_n([reconstruction_loss, lossL2])

			total_loss_summary = tf.summary.scalar("total_loss", total_loss)
			l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
			rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
			self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

			return total_loss

	def _network(self, data):
		with tf.variable_scope("Network", reuse=False):
			rnn_cell = tf.contrib.rnn.BasicLSTMCell(self._lstmParams.lstmSize())

			# generate prediction
			# dataset = tf.reshape(data, (-1, self._lstmParams.fftFreqBins(), 1))
			# print(dataset.shape)
			dataset = tf.split(data, 4, -3)

			with tf.variable_scope('first', reuse=False):
				layers_filters = self._weight_variable([1, 4, 2, 1])
				conv = np.empty([64, 0, 22, 1], dtype=np.float32)
				for time_step in dataset:
					inter_conv = tf.nn.conv2d(time_step, layers_filters, strides=(1, 1, 3, 1), padding="SAME")
					conv = tf.concat([conv, inter_conv], 1)

			dataset = tf.split(conv, 4, -3)

			with tf.variable_scope('second', reuse=False):
				layers_filters = self._weight_variable([1, 4, 1, 1])
				conv = np.empty([64, 0, 8, 1], dtype=np.float32)
				for time_step in dataset:
					inter_conv = tf.nn.conv2d(time_step, layers_filters, strides=(1, 1, 3, 1), padding="SAME")
					conv = tf.concat([conv, inter_conv], 1)

			dataset = tf.split(conv, 4, -3)

			with tf.variable_scope('third', reuse=False):
				layers_filters = self._weight_variable([1, 3, 1, 1])
				conv = np.empty([64, 0, 3, 1], dtype=np.float32)
				for time_step in dataset:
					inter_conv = tf.nn.conv2d(time_step, layers_filters, strides=(1, 1, 3, 1), padding="SAME")
					conv = tf.concat([conv, inter_conv], 1)

			dataset = tf.split(conv, 4, -3)

			with tf.variable_scope('fourth', reuse=False):
				layers_filters = self._weight_variable([1, 2, 1, 1])
				conv = np.empty([64, 0, 1, 1], dtype=np.float32)
				for time_step in dataset:
					inter_conv = tf.nn.conv2d(time_step, layers_filters, strides=(1, 1, 3, 1), padding="SAME")
					conv = tf.concat([conv, inter_conv], 1)

			conv = tf.reshape(conv, [64, 612])
			dataset = tf.split(conv, 4, -1)

			outputs, states = tf.nn.static_rnn(rnn_cell, dataset, dtype=tf.float32)
			# there are n_input outputs but
			# we only want the last output
			output = tf.matmul(outputs[-1], self._weight_variable(
				[self._lstmParams.lstmSize(), self._lstmParams.fftFreqBins()*2])) + self._bias_variable(
				[self._lstmParams.fftFreqBins()*2])
			return tf.reshape(output, [-1, self._lstmParams.fftFreqBins(), 2])


	def _weight_variable(self, shape):
		return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

	def _bias_variable(self, shape):
		return tf.get_variable('bias', shape, initializer=tf.contrib.layers.xavier_initializer())
