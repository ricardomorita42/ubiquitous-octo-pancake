# Idéia inicial: Pega um arquivo .csv e carrega os arquivos dentre dele para
# devolver um dict de strs. Os values de cada str será o feature de cada arquivo.
# Sobre padding: https://arxiv.org/pdf/1903.07288.pdf
# Necessário jdata:
# https://pypi.org/project/jdata/

import os
import csv
import pprint
import json

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

    assert os.path.isfile(self.fileName),"arquivo para importar não existe!"
    print("arquivo recebido:" + self.fileName)
    
    # 1o caso: Existe um arquivo já processado para este csv
    # Notar que no momento não checamos se o config atual é igual ao salvo!
    saved_file = self.fileName[:-4] + '_processed.json'
    if os.path.isfile(saved_file):
      print("Arquivo pré-processado encontrado. Carregando este arquivo...")
      self.preloadedFile = True
      self.datasetDict=jd.load(saved_file)

    # 2o. Caso: Não encontramos arquivo já processado.
    else:
      file = open(self.fileName)
      #print(type(file))

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
      
      #values estão como valor None neste momento (não são usadas)
      for key, value in self.getWholeDataset().items():
        feature = ap.wav2feature(key)
        self.setItem(key=key,value=feature)

      print("Salvando para disco...")
      self.save2file()

  def save2file(self, file_name=None):
    # Artificio para usar self como default value
    if file_name is None:
      file_name = self.fileName[:-4] + '_processed.json'
    
    jd.save(self.datasetDict,file_name)

    print('Salvo em ' + file_name)

  # Setters e getters básicos
  def getItem(self,key):
    if self.datasetDict[key] == None:
      return "None"
    else:
      return self.datasetDict[key]

  def setItem(self,key,value):
    self.datasetDict[key] = value

  def getWholeDataset(self):
    return self.datasetDict 








    

  