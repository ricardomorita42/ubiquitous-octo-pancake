if [ -z "$1" ]; then
    echo "Uso: ./filter_csv.sh pacientes.csv"
else
    if [ -f "$1" ]; then
        echo "Filtrando o CSV de pacientes: $1"
        python ./scripts/make_pacients.py $1 SPIRA_Dataset_V2/metadata.csv
        echo "Splitando em test e treino"
        python ./scripts/split_test_train.py SPIRA_Dataset_V2/metadata.csv
    else
        echo "Arquivo $1 não foi encontrado"
    fi
fi
