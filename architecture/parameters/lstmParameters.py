class LstmParameters(object):
	def __init__(self, lstmSize, signalLength, fftWindowLength, fftHopSize):
		self._signalLength = signalLength
		self._lstmSize = lstmSize
		self._fftWindowLength = fftWindowLength
		self._fftHopSize = fftHopSize

	def signalLength(self):
		return self._signalLength

	def lstmSize(self):
		return self._lstmSize

	def fftFreqBins(self):
		return self._fftWindowLength//2+1

	def fftWindowLength(self):
		return self._fftWindowLength

	def fftHopSize(self):
		return self._fftHopSize
