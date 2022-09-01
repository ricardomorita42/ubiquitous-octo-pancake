import os
import csv
import sys

def main():
    if len(sys.argv) != 3:
        print("Uso: python make_paients.py pacientes.csv filtered_pacientes.csv")
        return

    origin, destiny = sys.argv[1], sys.argv[2]

    if not os.path.isfile(origin):
        print("O arquivo passado não existe!")
        return

    with open(origin) as file:
        csvreader = csv.DictReader(file)

        with open(destiny, 'w') as new_file:
            fieldnames = ["audio_path", "sexo", "idade", "spO2"]
            writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in csvreader:
                # caminho do áudio, sexo, idade, saturação de O2
                audio_path = "SPIRA_Dataset_V2/pacientes/" + os.path.basename(row["palavras_opus"][:-5] + ".wav")

                if os.path.isfile(audio_path):
                    writer.writerow({
                        "audio_path": audio_path,
                        "sexo": row["sexo"],
                        "idade": row["idade"],
                        "spO2": row["oxigenacao"]
                        })

main()
