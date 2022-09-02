from sklearn.model_selection import train_test_split
import csv
import os
import argparse


def write_csv(filename, header, data):
    with open(filename, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(
        description='Split original csv in test and train')
    parser.add_argument('original_csv',
                        metavar='pacients.csv',
                        nargs=1,
                        type=str,
                        help="Original CSV filepath")
    args = parser.parse_args()

    csv_file = args.original_csv[0]
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
