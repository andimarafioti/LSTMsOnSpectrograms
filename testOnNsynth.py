from architecture.parameters.lstmParameters import LstmParameters
from architecture.simpleLSTMArchitecture import SimpleLSTMArchitecture
from system.lstmPreAndPostProcessor import LSTMPreAndPostProcessor
from system.lstmSystem import LSTMSystem

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "nsynth_"

batch_size = 64
outputWindowCount = 100
params = LstmParameters(lstmSize=512, batchSize=batch_size, signalLength=16384, fftWindowLength=256, fftHopSize=64,
						outputWindowCount=outputWindowCount)

aContextEncoderArchitecture = SimpleLSTMArchitecture(inputShape=(params.batchSize(),
																 params.fftFrames()-outputWindowCount,
																 params.fftFreqBins()), lstmParams=params)

aPreProcessor = LSTMPreAndPostProcessor(params)

aContextEncoderSystem = LSTMSystem(aContextEncoderArchitecture, batch_size, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("chopin_w16384_g8192_h1024.tfrecords", "chopin_w16384_g8192_h1024.tfrecords", 1e-3)
