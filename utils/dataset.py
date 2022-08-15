# Idéia inicial: Pega um arquivo .csv e carrega os arquivos dentre dele para
# devolver uma lista de strs. 
import os

class Dataset:
  def __init__(self, file_path, file_name):
    # Para evitar que '\0' vire null
    self.fileName = file_path + '/' + file_name
    
    assert os.path.isfile(self.fileName),"arquivo para importar não existe!"
    print("arquivo recebido:" + self.fileName)
    