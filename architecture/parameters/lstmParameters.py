class LstmParameters(object):
	def __init__(self, lstmSize, batchSize, signalLength, fftWindowLength, fftHopSize, outputWindowCount):
		self._signalLength = signalLength
		self._batchSize = batchSize
		self._lstmSize = lstmSize
		self._fftWindowLength = fftWindowLength
		self._fftHopSize = fftHopSize
		self._outputWindowCount = outputWindowCount

	def signalLength(self):
		return self._signalLength

	def batchSize(self):
		return self._batchSize

	def lstmSize(self):
		return self._lstmSize

	def fftFreqBins(self):
		return self._fftWindowLength//2+1

	def fftFrames(self):
		return (self._signalLength-self._fftWindowLength)/self._fftHopSize+1

	def fftWindowLength(self):
		return self._fftWindowLength

	def fftHopSize(self):
		return self._fftHopSize

	def outputWindowCount(self):
		return self._outputWindowCount
