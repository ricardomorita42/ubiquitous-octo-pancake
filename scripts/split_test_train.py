from re import X
from sklearn.model_selection import train_test_split
import sys
import csv
import os


def write_csv(filename, header, data):
    with open(filename, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(data)

def main():
    if len(sys.argv) != 2:
        print("Uso: python split_test_train.py pacientes.csv")
        return

    csv_file = sys.argv[1]
    x, y = [], []

    if not os.path.isfile(csv_file):
        print("O arquivo " + csv_file + " não foi encontrado")
        return

    # audio_path, sexo, idade, spO2
    with open(csv_file) as file:
        csv_reader = csv.reader(file)
        csv_header = next(csv_reader)

        for row in csv_reader:
            x.append(row)
            y.append(row[3])

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=0)

    root_dir = os.path.split(csv_file)[0]

    write_csv(root_dir + '/metadata_train.csv', csv_header, x_train)
    write_csv(root_dir + '/metadata_test.csv', csv_header, x_test)

main()
