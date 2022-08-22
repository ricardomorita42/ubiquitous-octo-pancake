# Idéia inicial: Pega um arquivo .csv e carrega os arquivos dentro dele para
# devolver um dict de strs. Os values de cada str serão o feature de cada arquivo.
# Sobre padding: https://arxiv.org/pdf/1903.07288.pdf
# Necessário jdata:
# https://pypi.org/project/jdata/

import os
import csv
# import pprint

import jdata as jd

class Dataset:
  def __init__(self, ap, file_path, file_name):
    # Para evitar que '\0' vire null
    self.filePath = file_path
    self.fileName = file_path + '/' + file_name

    self.ap = ap
    self.datasetHeader = []
    self.datasetDict = {}
    self.maxLength = 0
    self.preloadedFile = False

    assert os.path.isfile(self.fileName), "Arquivo do dataset não existe!"
    print("Arquivo recebido: " + self.fileName)

    saved_file = self.fileName[:-4] + "_processed.json"

    # Configuração do dataset atualmente é apenas o Audio Processor
    dataset_config = "dataset_config.json"

    if not os.path.isfile(saved_file) or not os.path.isfile(dataset_config) or \
      not self.cmpDictExcept(jd.load(dataset_config), self.ap.__dict__, ["feature", "max_length"]):
      # Arquivo pré-processado não foi encontrado ou as configurações eram diferentes

      file = open(self.fileName)
      csvreader = csv.reader(file)

      self.dataSetheader = next(csvreader)
      #print(self.header)

      # Quais atributos precisamos manter? Mudar aqui se quiser mais dados do csv
      for row in csvreader:
        fullPath = self.filePath + '/' + row[0]
        self.datasetDict[fullPath] = None
      #pprint.pprint(self.datasetDict)

      print("Descobrindo max length...")
      for key, value in self.getWholeDataset().items():
        # Internamente é registrado para cada chamada o max_length local
        ap.extractMaxLength(key)

      print("Max Length: " + str(ap.getMaxLength()))

      print("Calculando MFCCs...")
      # values estão como valor None neste momento (não são usadas)
      for key, value in self.getWholeDataset().items():
        feature = ap.wav2feature(key)
        self.setItem(key=key, value=feature)

      print("Salvando para disco...")
      self.save2file()
      print("Salvando arquivo de configuração: " + dataset_config)
      jd.save(self.ap.__dict__, dataset_config)
    else:
      # Existe um arquivo já processado e com essa configuração para este csv
      print("Arquivo pré-processado encontrado: " + saved_file + "\nCarregando este arquivo...")
      self.preloadedFile = True
      self.datasetDict = jd.load(saved_file)

  def save2file(self, file_name=None):
    # Artificio para usar self como default value
    if file_name is None:
      file_name = self.fileName[:-4] + "_processed.json"

    jd.save(self.datasetDict, file_name)

    print("Salvo em " + file_name)

  # Setters e getters básicos
  def getItem(self,key):
    if self.datasetDict[key] == None:
      return None
    else:
      return self.datasetDict[key]

  def setItem(self,key,value):
    self.datasetDict[key] = value

  def getWholeDataset(self):
    return self.datasetDict

  def cmpDictExcept(self, dict1, dict2, excluded_keys):
    """
    Compara dois dicionários, exceto pelas chaves passadas em excluded_keys
    """
    dct1 = {k: v for k, v in dict1.items() if k not in excluded_keys}
    dct2 = {k: v for k, v in dict2.items() if k not in excluded_keys}
    return dct1 == dct2
