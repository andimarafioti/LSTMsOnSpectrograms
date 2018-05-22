import tensorflow as tf
import numpy as np
from system.dnnSystem import DNNSystem
from utils.tfReader import TFReader


class LSTMSystem(DNNSystem):
	def __init__(self, architecture, batchSize, aPreProcessor, lstmParameters, name):
		self._aPreProcessor = aPreProcessor
		self._lstmParameters = lstmParameters
		self._windowSize = lstmParameters.signalLength()
		self._batchSize = batchSize
		self._audio = tf.placeholder(tf.float32, shape=(batchSize, self._windowSize), name='audio_data')
		self._inputAndTarget = aPreProcessor.inputAndTarget(self._audio)
		super().__init__(architecture, name)
		self._SNR = tf.reduce_mean(self._pavlovs_SNR(self._architecture.output(), self._architecture.target(), onAxis=[1]))

	def generate(self, STFT, length=100, model_num=None):
		with tf.Session() as sess:
			if model_num is not None:
				path = self.modelsPath(model_num)
			else:
				path = self.modelsPath()
			saver = tf.train.Saver()
			saver.restore(sess, path)
			print("Model restored.")
			sess.run([tf.local_variables_initializer()])

			inputShape = list(STFT.shape)
			outputShape = inputShape
			outputShape[0] = inputShape[0] + length
			spectrograms = np.zeros(outputShape, dtype=np.float32)
			spectrograms[:len(STFT)] = STFT
			for i in range(length):
				print(spectrograms.shape)
				input_data = spectrograms[-self._lstmParameters.countOfFrames():]
				feed_dict = {self._architecture.testInput(): input_data, self._architecture.isTraining(): False}
				nextSpectrogram = sess.run(self._architecture.generatedOutput(), feed_dict=feed_dict)
				print(nextSpectrogram.shape)
				spectrograms[len(STFT)+i] = nextSpectrogram
			return spectrograms

	def optimizer(self, learningRate):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			return tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self._architecture.loss())

	def _feedDict(self, data, sess, isTraining=True):
		net_input, net_target = sess.run(self._inputAndTarget,
										feed_dict={self._audio: data})
		return {self._architecture.input(): net_input, self._architecture.target(): net_target,
				self._architecture.isTraining(): isTraining}

	def _evaluate(self, summariesDict, feed_dict, validReader, sess):
		trainSNRSummaryToWrite = sess.run(summariesDict['train_SNR_summary'], feed_dict=feed_dict)

		try:
			audio = validReader.dataOperation(session=sess)
		except StopIteration:
			print("valid End of queue!")
			return [trainSNRSummaryToWrite]
		feed_dict = self._feedDict(audio, sess, False)
		validSNRSummary = sess.run(summariesDict['valid_SNR_summary'], feed_dict)

		return [trainSNRSummaryToWrite, validSNRSummary]

	def _loadReader(self, dataPath):
		return TFReader(dataPath, self._windowSize, batchSize=self._batchSize, capacity=int(2e5), num_epochs=400)

	def _evaluationSummaries(self):
		summaries_dict = {'train_SNR_summary': tf.summary.scalar("training_SNR", self._SNR),
						  'valid_SNR_summary': tf.summary.scalar("validation_SNR", self._SNR)}
		return summaries_dict

	def _squaredEuclideanNorm(self, tensor, onAxis=[1, 2, 3]):
		squared = tf.square(tensor)
		summed = tf.reduce_sum(squared, axis=onAxis)
		return summed

	def _log10(self, tensor):
		numerator = tf.log(tensor)
		denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
		return numerator / denominator

	def _pavlovs_SNR(self, y_orig, y_inp, onAxis=[1, 2, 3]):
		norm_y_orig = self._squaredEuclideanNorm(y_orig, onAxis)
		norm_y_orig_minus_y_inp = self._squaredEuclideanNorm(y_orig - y_inp, onAxis)
		return 10 * self._log10(norm_y_orig / norm_y_orig_minus_y_inp)
