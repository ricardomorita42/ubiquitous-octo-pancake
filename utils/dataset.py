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

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Dataset(Dataset):
    def __init__(self, ap, file_path):
        self.ap = ap
        self.filePath = file_path
        self.datasetHeader = []
        self.datasetDict = {}
        self.datasetRefs = []
        self.maxLength = 0

        assert os.path.isfile(self.filePath), "Arquivo do dataset não existe!"
        print("Arquivo recebido:", self.filePath)

        saved_file = os.path.splitext(self.filePath)[0] + "_processed.json"

        # Configuração do dataset atualmente é apenas o Audio Processor
        dataset_config = os.path.splitext(self.filePath)[0] + "_dataset_config.json"

        if not os.path.isfile(saved_file) or not os.path.isfile(dataset_config) or \
          not cmpDictExcept(jd.load(dataset_config)["ap"], self.ap.__dict__, ["max_length"]):
            # Arquivo pré-processado não foi encontrado ou as configurações eram diferentes

            with open(self.filePath) as file:
                csvreader = csv.DictReader(file)

                self.datasetHeader = csvreader.fieldnames

                # Quais atributos precisamos manter? Mudar aqui se quiser mais dados do csv
                basePath = os.path.dirname(self.filePath)
                for row in csvreader:
                    fullPath = os.path.join(basePath, row["audio_path"])
                    self.setItem(key=fullPath, value=[row["sexo"], row["idade"], row["spO2"]])
                #pprint.pprint(self.datasetDict)
            
            max_length_pbar = tqdm(total=len(self.datasetDict))
            max_length_pbar.set_description_str("Descobrindo max length")
            for key in self.datasetDict:
                # Internamente é registrado para cada chamada o max_length local
                ap.extractMaxLength(key)
                max_length_pbar.update(1)
            max_length_pbar.close()
            
            #ap.max_length = 10000
            print("Max Length:", str(ap.getMaxLength()))

            # values estão como valor None neste momento (não são usadas)
            mfcc_pbar = tqdm(total=len(self.datasetDict))
            mfcc_pbar.set_description_str("Calculando MFCCs")
            for key, val in self.datasetDict.items():
                feature = ap.wav2featureWindowing(key)
                self.setItem(key=key, value=[feature, *val])
                
                for f in feature:
                    ref = [f,int(val[2])]
                    self.datasetRefs.append(ref)
                mfcc_pbar.update(1)
            mfcc_pbar.close()

            print("Salvando para disco...")
            self.save2file()
            print("Salvando arquivo de configuração: " + dataset_config)
            jd.save({"ap": self.ap.__dict__, "datasetHeader": self.datasetHeader}, dataset_config)
        else:
            # Existe um arquivo já processado e com essa configuração para este csv
            print("Arquivo pré-processado encontrado: " + saved_file + "\nCarregando este arquivo...")
            self.datasetDict = jd.load(saved_file)
            self.ap.max_length = jd.load(dataset_config)["ap"]["max_length"]
            self.datasetHeader = jd.load(dataset_config)["datasetHeader"]

    def save2file(self, file_name=None):
        # Artificio para usar self como default value
        if file_name is None:
            file_name = os.path.splitext(self.filePath)[0] + "_processed.json"

        jd.save(self.datasetDict, file_name)

        print("Salvo em " + file_name)

    # Setters e getters básicos
    def getItem(self, key):
        if key in self.datasetDict:
            return self.datasetDict[key]
        else:
            return None

    def setItem(self, key, value):
        self.datasetDict[key] = value

    def getWholeDataset(self):
        return self.datasetDict

    # Funções mágicas

    def __len__(self):
        counter = 0

        for val in self.datasetDict.values():
            counter += len(val[0])

        return counter

    def __getitem__(self, idx):
        return self.datasetRefs[0], self.datasetRefs[1]

        

def train_dataloader(c, ap):
    return DataLoader(dataset=Dataset(ap, c.dataset["train_csv"]),
                      pin_memory=True,
                      num_workers=c.train_config["num_workers"],
                      batch_size=c.train_config["batch_size"],
                      collate_fn=own_collate_fn,
                      drop_last=True,
                      sampler=None)

def test_dataloader(c, ap):
    return DataLoader(dataset=Dataset(ap, c.dataset["test_csv"]),
                      pin_memory=True,
                      num_workers=c.train_config["num_workers"],
                      batch_size=c.train_config["batch_size"],
                      collate_fn=own_collate_fn,
                      drop_last=True,
                      sampler=None)

def own_collate_fn(batch):
    features, targets = [], []

    for (feature, target) in batch:
        features.append(torch.tensor(feature.transpose()))
        targets.append([target])

    targets = torch.tensor(targets, dtype=torch.float)
    features = pad_sequence(features, batch_first=True, padding_value=0)

    return features, targets
    