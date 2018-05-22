from architecture.parameters.lstmParameters import LstmParameters
from architecture.simpleLSTMArchitecture import SimpleLSTMArchitecture
from system.lstmPreAndPostProcessor import LSTMPreAndPostProcessor
from system.lstmSystem import LSTMSystem

sessionsName = "LSTM_test_2_"

params = LstmParameters(lstmSize=512, signalLength=5120, fftWindowLength=128, fftHopSize=32)
batch_size = 1

aContextEncoderArchitecture = SimpleLSTMArchitecture(inputShape=(batch_size, 612, params.fftFreqBins()), lstmParams=params)

aPreProcessor = LSTMPreAndPostProcessor(params)

aContextEncoderSystem = LSTMSystem(aContextEncoderArchitecture, batch_size, aPreProcessor, sessionsName)

aContextEncoderSystem.train("fake_w5120_g1024_h512.tfrecords", "fake_w5120_g1024_h512.tfrecords", 1e-3)