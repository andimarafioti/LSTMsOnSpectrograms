from architecture.parameters.lstmParameters import LstmParameters
from architecture.simpleLSTMArchitecture import SimpleLSTMArchitecture
from system.lstmSystem import LSTMSystem

import os

from system.realImagLSTMPreAndPostProcessor import RealImagLSTMPreAndPostProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_RealImag_"

batch_size = 64
params = LstmParameters(lstmSize=512, batchSize=batch_size, signalLength=5120, fftWindowLength=128, fftHopSize=32,
						countOfFrames=4)

aContextEncoderArchitecture = SimpleLSTMArchitecture(inputShape=(params.batchSize(),
																 params.fftFrames()-1,
																 params.fftFreqBins()), lstmParams=params)

aPreProcessor = RealImagLSTMPreAndPostProcessor(params)

aContextEncoderSystem = LSTMSystem(aContextEncoderArchitecture, batch_size, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("fake_w5120_g1024_h512.tfrecords", "fake_w5120_g1024_h512.tfrecords", 1e-3)
