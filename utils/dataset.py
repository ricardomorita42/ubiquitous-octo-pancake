# Idéia inicial: Pega um arquivo .csv e carrega os arquivos dentre dele para
# devolver um dict de strs. Os values de cada str será o feature de cada arquivo.
# Sobre padding: https://arxiv.org/pdf/1903.07288.pdf

import os
import csv
import pprint

class Dataset:
  def __init__(self, file_path, file_name):
    # Para evitar que '\0' vire null
    self.fileName = file_path + '/' + file_name
    self.datasetHeader = []
    self.datasetList = {}

    assert os.path.isfile(self.fileName),"arquivo para importar não existe!"
    print("arquivo recebido:" + self.fileName)

    file = open(self.fileName)
    #print(type(file))

    csvreader = csv.reader(file)

    self.dataSetheader = next(csvreader)
    #print(self.header)
    
    # Quais atributos precisamos manter? Mudar aqui se quiser mais dados do csv
    for row in csvreader:
      self.datasetList[row[0]] = None
    pprint.pprint(self.datasetList)

  
  # Setters e getters básicos
  def getItem(self,key):
    return self.datasetList[key]

  def setItem(self,key,value):
    self.datasetList[key] = value

  def getWholeDataset(self):
    return self.datasetList 








    

  