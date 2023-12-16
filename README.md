# Detection of letters from the American Sign Alphabet

Este repositório contém o código e os recursos necessários ao projeto de Visão Computacional com detecção de linguagem de sinais.

## Descrição

O objetivo deste projeto é desenvolver um sistema de detecção e reconhecimento de linguagem de sinais em tempo real utilizando técnicas de visão computacional. Três modelos foram treinados utilizando datasets disponíveis no Kaggle:

1. [ASL Alphabet Test Dataset](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test)
2. [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
3. [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

O código `detection_v3.py` representa a versão mais avançada do modelo de detecção, treinado utilizando um conjunto combinado desses datasets para o segundo modelo, recomenda-se usar o terceiro modelo, pois apresenta melhor acurácia apesar da dificuldade da precisão da deteção.

## Notebooks Jupyter
Três modelos foram treinados e estão disponiveis no `Vision.ipynb`. Por motivos de segurança, as credenciais do Kaggle não foram fornecidas diretamente no repositório.

## Estrutura do Projeto

- `detection_v3.py`: Código principal para detecção de linguagem de sinais.
- `index.html`: Página HTML para a interface do usuário.
- `templates/`: Pasta contendo o arquivo HTML usados pelo Flask.

## Pré-requisitos

- Python 3.x
- Bibliotecas Python: Flask, OpenCV, TensorFlow, NumPy, Matplotlib 

Instale as bibliotecas necessárias utilizando:

- `pip install opencv-python tensorflow numpy flask matplotlib`

## Execução
1. Baixe o modelo treinado de detecção de linguagem de sinais
2. Coloque o modelo na mesma pasta que o arquivo detection_v3.py.
3. Execute o aplicativo usando o comando: `python detection_v3.py`
4. Abra o navegador que o código irá disponibilizar e acesse para visualizar o aplicação.

## Recomendações e Observações
Dado o problema na questão do dataset, o melhor modelo que é o terceiro, mesmo usando Data Augmentation, ficar muito difícil de detectar por conta que o sinal precisa ser igual ou muito paraceido ao que é o dataset. Em trabalhos futuros, recomenda-se um sistema de coleta volutário de sinais afim de ter um grande dataset com uma variedade de dados e tonalidades diferentes, pode ser feito usando IA Generativa também. Devido a limitações de upload de arquivo do Github, o segundo modelo não pode ser upado, o tamanho dele excede o permitido pela plataforma, aqueles que quiserem o segundo modelo para o teste, façam um merge em alguma coisa do código que irei enviar o segundo modelo.
