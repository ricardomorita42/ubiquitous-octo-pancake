# Idéia inicial: Pega um arquivo .csv e carrega os arquivos dentre dele para
# devolver um dict de strs. Os values de cada str será o feature de cada arquivo.
# Sobre padding: https://arxiv.org/pdf/1903.07288.pdf

import os
import csv
import pprint

class Dataset:
  def __init__(self, file_path, file_name):
    # Para evitar que '\0' vire null
    self.filePath = file_path
    self.fileName = file_path + '/' + file_name
    self.datasetHeader = []
    self.datasetDict = {}
    self.maxLength = 0

    assert os.path.isfile(self.fileName),"arquivo para importar não existe!"
    print("arquivo recebido:" + self.fileName)

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








    

  