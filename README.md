# TCC Ricardo e Natália IME 2022
README ainda em desenvolvimento.

## Tema
Desenvolvimento de biomarcadores a partir de voz: análise de áudio para detecção do estado de saúde de pacientes admitidos no HC.

## Supervisor
Marcelo Finger (mfinger at ime.usp.br)

## Alunos
Ricardo Mikio Morita (ricardo.morita at usp.br)
Natália Hitomi Koza (natalia.koza at usp.br)

## Proposta
- A ideia do TCC é realizar a previsão de SpO2 (saturação de O2) do paciente através da análise de áudio do mesmo utilizando métodos de inteligência artificial.
- Dados obtidos foram inicialmente de pacientes com suspeita de COVID do HC. Um estudo anterior conseguiu acurácia ao redor de 91% para pacientes <= 92% SpO2. 
- Nossa proposta é tentar estimar SpO2 em pacientes através de análise da voz do mesmo. Qual a precisão e desvio padrão da análise?- 
Por quê estudar SpO2? Pacientes com COVID possuem um sintoma de “hipóxia silenciosa”, o que motivou o estudo anterior. Mas se fosse possível desenvolver um biomarcador que analise o áudio, este seria barato e uma possível ferramenta para Médicos (biomarcador não pode diagnosticar mas pode auxiliar no diagnóstico).
- Dados disponíveis:
  - Áudio do paciente (gravado com o quê?, quais as condições do paciente na hora da gravação?)
  - Freq. Cardíaca (bpm) (medido com o quê? oxímetro)
  - Saturação O2 (oxímetro)
  - Conhecimentos importantes para o estudo:
  - Processamento de linguagem natural (PLN)
  - Tratamento de sinais (MFCC e espectrograma), Transformada de Fourier
