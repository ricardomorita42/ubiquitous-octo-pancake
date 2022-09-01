if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Uso: ./filter_csv.sh pacientes.csv filtered_pacientes.csv"
else
    if [ -f "$1" ]; then
        echo "Filtrando o CSV de pacientes: $1"
        python ./scripts/make_pacients.py $1 $2
    else
        echo "Arquivo $1 não foi encontrado"
    fi
fi
