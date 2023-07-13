# Avaliação de Modelos: KNN vs. Regressão Linear

Este código realiza uma avaliação comparativa entre os modelos de Regressão KNN (K-Nearest Neighbors) e Regressão Linear usando um conjunto de dados de notas de exames. Ele segue as etapas definidas para treinar, validar e testar os modelos, além de repetir o processo 30 vezes para obter estimativas do erro médio de cada modelo.

## Pré-requisitos

Antes de executar o código, certifique-se de ter instalado as seguintes bibliotecas do Python:

- pandas
- scikit-learn (sklearn)
- numpy

Você pode instalá-las usando o gerenciador de pacotes `pip` da seguinte maneira:

```
pip install pandas scikit-learn numpy
```

## Como executar o código

1. Clone ou faça o download deste repositório.

2. Navegue até o diretório onde o código está localizado.

3. Execute o código em um ambiente Python (por exemplo, Jupyter Notebook, Google Colab ou seu ambiente de desenvolvimento preferido).

4. O código carrega um conjunto de dados contendo notas de exames de alunos. Ele realiza as seguintes etapas:

    - Pré-processamento dos dados, transformando as variáveis categóricas em numéricas.
    - Divisão dos dados em treino (70%), validação (10%) e teste (20%).
    - Ajuste dos modelos KNN e Regressão Linear.
    - Uso do conjunto de validação para escolher os melhores parâmetros dos modelos (não implementado neste código).
    - Avaliação dos modelos no conjunto de validação.
    - Auxílio dos modelos na base de treino + validação.
    - Teste dos modelos no conjunto de teste.
    - Cálculo do erro médio para cada modelo.
    - Cálculo do intervalo de confiança de 95% para o erro médio de cada modelo.
    - Impressão dos resultados.

5. O código repetirá o processo de ajuste e avaliação dos modelos 30 vezes para obter estimativas mais robustas do erro médio.

## Resultados

Após a execução do código, os resultados serão impressos na saída. Serão fornecidas as seguintes informações para cada modelo:

- Estimativa pontual do erro médio: a média dos erros obtidos em cada repetição.
- Intervalo de confiança (95%) para o erro médio: os valores mínimo e máximo do erro médio, que representam o intervalo de confiança de 95% calculado a partir das repetições.

Além disso, o código concluirá qual modelo apresentou o menor erro médio.

## Personalização do código

O código fornecido é um exemplo básico para realizar a avaliação comparativa entre os modelos KNN e Regressão Linear. É possível personalizar o código de acordo com suas necessidades, como a seleção de diferentes variáveis de resposta, a modificação dos parâmetros dos modelos ou a implementação de técnicas de otimização de hiperparâmetros.

## Contribuição

Sinta-se à vontade para contribuir para este projeto fazendo sugestões, identificando problemas ou abrindo pull requests.
