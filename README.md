# TCC Ricardo e Natália IME 2022
README ainda em desenvolvimento.

## Tema
Desenvolvimento de biomarcadores a partir de voz: análise de áudio para previsão do nível de saturação de oxigênio no sangue (SpO2).

## Supervisor
- Marcelo Finger (mfinger at ime.usp.br)

## Alunos
- Ricardo Mikio Morita (ricardo.morita at usp.br)
- Natália Hitomi Koza (natalia.koza at usp.br)

## Proposta
- Realizar a previsão de SpO2 (saturação de O2) através da análise de um trecho de fala, utilizando métodos de inteligência artificial.
- Um estudo anterior conseguiu acurácia ao redor de 91% para pacientes detectar nível de SpO2 inferior a 92%
- Por quê estudar SpO2?
  - Pacientes com COVID possuem um sintoma de “hipóxia silenciosa”, o que motivou o estudo anterior.
  - Mas se fosse possível desenvolver um biomarcador que analise o áudio e atravésde um smartphone, este seria barato e uma possível ferramenta para médicos (biomarcador não pode diagnosticar mas pode auxiliar no diagnóstico)
  - Pode ajudar na indicação de possíveis problemas de insuficiência respiratória, não só COVID-19.


## Dados
Os dados foram obtidos de pacientes com suspeita de COVID no HC, e estão disponíveis sobre a licença CC BY-SA 4.0 [neste link do Github](https://github.com/Edresson/SPIRA-ACL2021):
  - Áudio do paciente: gravado com o celular
  - Frequência cardíaca (bpm): medida com o oxímetro
  - Saturação de O2 (SpO2): medida com o oxímetro
- Conhecimentos importantes para o estudo:
  - Processamento de linguagem natural (PLN)
  - Tratamento de sinais (MFCC e espectrograma), Transformada de Fourier


## Detalhes de implementação
A divisão dos pacientes em teste/treinamento/validação estão na pasta SPIRA_Dataset_V2/. Estes ainda não estão processados para serem usados, usando o script em scripts/make_pacients.py podemos gerar os arquivos processados.

Para rodar o código:
1. Criar um arquivo de configuração ou usar um de experiments/configs/
2. rodar train.py ou test.py de acordo com a operação desejada.

Exemplos de como invocar testes/treinamentos podem ser vistos nos scripts run_train.sh ou run_test.sh

	
