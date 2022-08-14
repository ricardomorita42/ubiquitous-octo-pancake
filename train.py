import argparse

from utils.audio_processor import AudioProcessor
from utils.generic import load_config

if __name__ == '__main__':
  '''
  Exemplo de uso: python train.py -c experiments/configs/exp-1.1.json
  '''

  # Converte e carrega arquivo json com dados do experimento
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config_path', type=str, required=True,
                      help="json file with configurations")
  args = parser.parse_args()
  c = load_config(args.config_path)

  ap = AudioProcessor(**c.audio)

  # audio_path = "resultado1.wav"
  # ap.wav2feature(audio_path)
  # ap.graph_feature()
