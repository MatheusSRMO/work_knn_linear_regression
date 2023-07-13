import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# Carregar o conjunto de dados
data_url = "exams.csv"
data_url = "http://roycekimmons.com/system/generate_data.php?dataset=exams&n=1000"
df = pd.read_csv(data_url)

categorical = df.drop(['math score', 'reading score', 'writing score'], axis=1)
numerical = df[['math score', 'reading score', 'writing score']]

df1 = categorical.apply(lambda x: pd.factorize(x)[0])

data = pd.concat([df1, numerical], axis=1, ignore_index=True)

new_columns_name = {
    0: 'gender',
    1: 'race',
    2: 'parent education',
    3: 'lunch',
    4: 'preparation tests',
    5: 'math',
    6: 'reading',
    7: 'writing',
}
data = data.rename(columns=new_columns_name)

# Variável resposta: math score
response_variable = 'math'

# Definir número de repetições
num_repeats = 30

# Listas para armazenar os resultados
knn_errors = []
linear_regression_errors = []

for _ in range(num_repeats):
    # Dividir os dados em treino (70%), validação (10%) e teste (20%)
    train_data, test_data = train_test_split(data, test_size=0.3)
    val_data, test_data = train_test_split(test_data, test_size=0.66)
    
    # Separar as variáveis covariáveis e a variável resposta
    X_train = train_data.drop(response_variable, axis=1)
    y_train = train_data[response_variable]
    
    X_val = val_data.drop(response_variable, axis=1)
    y_val = val_data[response_variable]
    
    X_test = test_data.drop(response_variable, axis=1)
    y_test = test_data[response_variable]
    
    # Ajustar o modelo KNN
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train, y_train)
    
    # Ajustar o modelo de regressão linear
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Usar conjunto de validação para escolher os melhores parâmetros dos modelos (não implementado)
    
    # Avaliar os modelos no conjunto de validação
    knn_predictions_val = knn_model.predict(X_val)
    linear_predictions_val = linear_model.predict(X_val)
    
    knn_error_val = mean_absolute_error(y_val, knn_predictions_val)
    linear_error_val = mean_absolute_error(y_val, linear_predictions_val)
    
    # Auxiliar o modelo na base de treino + validação
    knn_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    linear_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    # Testar os modelos no conjunto de teste
    knn_predictions_test = knn_model.predict(X_test)
    linear_predictions_test = linear_model.predict(X_test)
    
    knn_error_test = mean_absolute_error(y_test, knn_predictions_test)
    linear_error_test = mean_absolute_error(y_test, linear_predictions_test)
    
    # Armazenar os erros dos modelos
    knn_errors.append(knn_error_test)
    linear_regression_errors.append(linear_error_test)

# Calcular a estimativa pontual do erro médio para o KNN e para a regressão linear
knn_mean_error = np.mean(knn_errors)
linear_regression_mean_error = np.mean(linear_regression_errors)

# Calcular o intervalo de confiança de 95% para o erro médio do KNN e para a regressão linear
knn_error_interval = np.percentile(knn_errors, [2.5, 97.5])
linear_regression_error_interval = np.percentile(linear_regression_errors, [2.5, 97.5])

# Imprimir os resultados
print("KNN:")
print("Estimativa pontual do erro médio:", knn_mean_error)
print("Intervalo de confiança (95%) para o erro médio:", knn_error_interval)
print()

print("Regressão Linear:")
print("Estimativa pontual do erro médio:", linear_regression_mean_error)
print("Intervalo de confiança (95%) para o erro médio:", linear_regression_error_interval)
print()

# Comparar os modelos
if knn_mean_error < linear_regression_mean_error:
    print("O modelo KNN apresentou um menor erro médio.")
elif linear_regression_mean_error < knn_mean_error:
    print("O modelo de Regressão Linear apresentou um menor erro médio.")
else:
    print("Ambos os modelos apresentaram o mesmo erro médio.")
