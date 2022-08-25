# Idéia inicial: Pega um arquivo .csv e carrega os arquivos dentro dele para
# devolver um dict de strs. Os values de cada str serão o feature de cada arquivo.
# Sobre padding: https://arxiv.org/pdf/1903.07288.pdf
# Necessário jdata:
# https://pypi.org/project/jdata/

import os
import csv
# import pprint

from tqdm import tqdm
from utils.generic import cmpDictExcept

import jdata as jd

class Dataset:
    def __init__(self, ap, file_path):
        self.ap = ap
        self.filePath = file_path
        self.datasetHeader = []
        self.datasetDict = {}
        self.maxLength = 0

        assert os.path.isfile(self.filePath), "Arquivo do dataset não existe!"
        print("Arquivo recebido:", self.filePath)

        saved_file = self.filePath[:-4] + "_processed.json"

        # Configuração do dataset atualmente é apenas o Audio Processor
        dataset_config = self.filePath[:-4] + "_dataset_config.json"

        if not os.path.isfile(saved_file) or not os.path.isfile(dataset_config) or \
          not cmpDictExcept(jd.load(dataset_config), self.ap.__dict__, ["feature", "max_length"]):
            # Arquivo pré-processado não foi encontrado ou as configurações eram diferentes

            file = open(self.filePath)
            csvreader = csv.reader(file)

            self.dataSetheader = next(csvreader)
            #print(self.header)

            # Quais atributos precisamos manter? Mudar aqui se quiser mais dados do csv
            basePath = os.path.dirname(self.filePath)
            for row in csvreader:
                fullPath = basePath + '/' + row[0]
                self.datasetDict[fullPath] = None
            #pprint.pprint(self.datasetDict)

            print("Descobrindo max length...")
            pbar = tqdm(total=len(self.datasetDict))
            for key, value in self.datasetDict.items():
                # Internamente é registrado para cada chamada o max_length local
                ap.extractMaxLength(key)
                pbar.update(1)
            pbar.close()

            print("Max Length: " + str(ap.getMaxLength()))

            print("Calculando MFCCs...")
            # values estão como valor None neste momento (não são usadas)
            pbar = tqdm(total=len(self.datasetDict))
            for key, value in self.datasetDict.items():
                feature = ap.wav2feature(key)
                self.setItem(key=key, value=feature)
                pbar.update(1)
            pbar.close()

            print("Salvando para disco...")
            self.save2file()
            print("Salvando arquivo de configuração: " + dataset_config)
            jd.save(self.ap.__dict__, dataset_config)
        else:
            # Existe um arquivo já processado e com essa configuração para este csv
            print("Arquivo pré-processado encontrado: " + saved_file + "\nCarregando este arquivo...")
            self.datasetDict = jd.load(saved_file)

    def save2file(self, file_name=None):
        # Artificio para usar self como default value
        if file_name is None:
            file_name = self.filePath[:-4] + "_processed.json"

        jd.save(self.datasetDict, file_name)

        print("Salvo em " + file_name)

    # Setters e getters básicos
    def getItem(self, key):
        if self.datasetDict[key] == None:
            return None
        else:
            return self.datasetDict[key]

    def setItem(self, key, value):
        self.datasetDict[key] = value

    def getWholeDataset(self):
        return self.datasetDict


    # Funções mágicas

    def __len__(self):
        return len(self.datasetDict)
