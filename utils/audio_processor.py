# requirements
# pip install librosa matplotlib
#
# https://stackoverflow.com/questions/44473110/compute-mfcc-using-librosa
# https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
# https://librosa.org/doc/main/auto_examples/plot_display.html
# https://stackoverflow.com/questions/54160128/feature-extraction-using-mfcc
# https://stackoverflow.com/questions/46031397/using-librosa-to-plot-a-mel-spectrogram
# https://librosa.org/doc/0.9.1/generated/librosa.stft.html
# https://github.com/librosa/librosa/issues/1251
# https://stackoverflow.com/questions/53925401/difference-between-mel-spectrogram-and-an-mfcc
# https://medium.com/analytics-vidhya/simplifying-audio-data-fft-stft-mfcc-for-machine-learning-and-deep-learning-443a2f962e0e
# https://stackoverflow.com/questions/52232839/understanding-the-output-of-mfcc

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Nesta implementação estamos sempre retornando mfcc em vez de retornar
# spectrograma ou melspectogram. Necessário adaptar feature caso esta
# situação mude.

class AudioProcessor:
  # Adicionar mais parâmetros conforme necessário
  def __init__(self, sr, hop_length, win_length,
               n_fft, n_mfcc, n_mels):
    self.sr = sr
    self.n_mfcc = n_mfcc
    self.hop_length = hop_length
    self.win_length = win_length
    self.n_fft = n_fft
    self.n_mels = n_mels
    self.feature = 0

  def wav2feature(self,audio_path):
    '''
    Retorna um ndarray que calcula Mel-frequency cepstral coefficents a partir
    de um .wav. Parâmetros adicionais para alterar o spectrogram podem ser
    passados através de **kwargs
    '''

    y, sr = librosa.load(audio_path)
    #y: np.ndarray that represents audio time series.
    #sr:  number > 0 [scalar] that represents sampling rate of y

    # Reamostra o áudio com a taxa de amostragem do experimento
    if sr != self.sr:
      y = librosa.resample(y=y, orig_sr=sr, target_sr=self.sr)

    # Extraindo mfcc
    # (1) MFCC is based on short-time Fourier transform (STFT), n_fft, hop_length, win_length
    # and window are the parameters for STFT.
    self.feature = librosa.feature.mfcc(y=y, sr=self.sr, hop_length=self.hop_length,
                   win_length=self.win_length, n_fft=self.n_fft, n_mfcc=self.n_mfcc,
                   n_mels=self.n_mels)
    return self.feature

  # Para debug
  def graph_feature(self):
    '''
    Função para debug, plota o gráfico contendo as features extraídas do áudio
    usando MFCC
    '''

    plt.figure(figsize=(10,4))
    librosa.display.specshow(self.feature, sr=self.sr, x_axis ='time')
    plt.show()

if __name__ == "__main__":
  '''
  # Todos os parâmetros do github da função da pasta utils do github do Edresson
  feature = "mfcc"
  num_mels = 40
  num_mfcc = 40
  log_mels = False
  mel_fmin = 0.0
  mel_fmax = 8000.0
  sample_rate = 16000
  normalization = True
  num_freq = 601

  hop_length = 160
  win_length = 400
  n_fft = 1200
'''

  print(__file__ + " invocado")
