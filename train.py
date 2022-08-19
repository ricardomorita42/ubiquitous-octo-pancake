import argparse
from unittest.util import _MAX_LENGTH

from utils.audio_processor import AudioProcessor
from utils.dataset import Dataset
from utils.generic import load_config

import random

if __name__ == '__main__':
  '''
  Exemplo de uso: python train.py -c experiments/configs/exp-1.1.json
  '''

  # Converte e carrega arquivo json com dados do experimento
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config_path', type=str,
                      default="experiments/configs/exp-1.1.json",
                      help="json file with configurations")
  args = parser.parse_args()
  c = load_config(args.config_path)

  ap = AudioProcessor(**c.audio)
  d = Dataset(**c.dataset)

  ### Candidato para uma nova biblioteca ###
  print("Descobrindo max length...")
  for key, value in d.getWholeDataset().items():
    #print(key + ": " + str(value))
    
    # Internamente é registrado para cada chamada o max_length local
    ap.extractMaxLength(key)

  print("Max Length: " + str(ap.getMaxLength()))

  #audio_path = random.choice(list(d.getWholeDataset().keys()))
  #ap.wav2feature(audio_path)
  #ap.graphFeature()

  print("Calculando MFCCs...")
  for key, value in d.getWholeDataset().items():
    feature = ap.wav2feature(key)
    d.setItem(key=key,value=feature)

  print("Imprimindo um exemplo")
  audio_path,feature = random.choice(list(d.getWholeDataset().items()))
  ap.graphFeature(feature)

  #audio_path = "SPIRA_Dataset_V2\controle\\0a2d6271-846b-4157-a784-b5fa2d93d2f9_1.wav"
  #ap.wav2feature(audio_path)
  #ap.graph_feature()
