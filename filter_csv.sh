echo "Filtrando somente dados de pacientes"

cat ./SPIRA_Dataset_V2/metadata_train.csv | grep ^pacientes > ./SPIRA_Dataset_V2/filtered_metadata_train.csv

cat ./SPIRA_Dataset_V2/metadata_eval.csv | grep ^pacientes > ./SPIRA_Dataset_V2/filtered_metadata_eval.csv

cat ./SPIRA_Dataset_V2/metadata_test.csv | grep ^pacientes > ./SPIRA_Dataset_V2/filtered_metadata_test.csv

echo "Terminou de filtrar"
