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
  d = Dataset(ap, **c.dataset)

  # Deixando esta parte executando para que se possa checar o funcionamento...
  print("Imprimindo um exemplo")
  audio_path,feature = random.choice(list(d.getWholeDataset().items()))
  print("Exemplo: " + audio_path)
  ap.graphFeature(feature)