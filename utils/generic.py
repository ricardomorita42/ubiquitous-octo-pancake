import re
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def cmpDictExcept(dict1, dict2, excluded_keys):
    """
  Compara dois dicionários, exceto pelas chaves passadas em excluded_keys
  """
    dct1 = {k: v for k, v in dict1.items() if k not in excluded_keys}
    dct2 = {k: v for k, v in dict2.items() if k not in excluded_keys}
    return dct1 == dct2

# adapted from https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, inp):
        '''
        Forward pass of the function.
        '''
        return inp * torch.tanh(F.softplus(inp))
