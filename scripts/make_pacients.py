import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(
        description='Filter original csv exported from db')
    parser.add_argument('origin',
                        metavar='original.csv',
                        nargs=1,
                        type=argparse.FileType('r'),
                        help="Original CSV filepath")
    parser.add_argument('destiny',
                        metavar='filtered.csv',
                        nargs='?',
                        type=argparse.FileType('w'),
                        default='SPIRA_Dataset_V2/metadata.csv',
                        help="Filtered file output name")
    args = parser.parse_args()

    fieldnames = ["audio_path", "sexo", "idade", "spO2"]
    csv_writer = csv.DictWriter(args.destiny, fieldnames=fieldnames)
    csv_writer.writeheader()

    csv_reader = csv.DictReader(args.origin[0])
    for row in csv_reader:
        # caminho do áudio, sexo, idade, saturação de O2
        audio_path = "pacientes/" + os.path.basename(
            row["palavras_opus"][:-5] + ".wav")

        if os.path.isfile("SPIRA_Dataset_V2/" + audio_path):
            csv_writer.writerow({
                "audio_path": audio_path,
                "sexo": row["sexo"],
                "idade": row["idade"],
                "spO2": row["oxigenacao"]
            })


main()
