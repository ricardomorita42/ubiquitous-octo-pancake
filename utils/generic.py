import re
import json

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def load_config(config_path):
  '''
  Converte arquivo .json com configurações para os experimentos em um
  dicionário
  '''

  # Autor: Edresson Casanova
  config = AttrDict()
  with open(config_path, "r") as f:
      input_str = f.read()
  input_str = re.sub(r'\\\n', '', input_str)
  input_str = re.sub(r'//.*\n', '\n', input_str)
  data = json.loads(input_str)
  config.update(data)
  return config
