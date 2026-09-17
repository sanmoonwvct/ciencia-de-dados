import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Geração da massa de dados sintética
X, y = make_classification(
    n_samples=500,        # Total de exemplos
    n_features=4,         # Número de atributos (variáveis continuas)
    n_informative=3,      # Atributos com relevância para a predição
    n_redundant=1,        # Atributos redundantes
    n_classes=2,          # Classificação binária (0 ou 1)
    random_state=42
)

# 2. Divisão entre treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicialização e treinamento do modelo Naïve Bayes Gaussiano
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Avaliação do modelo
y_pred = model.predict(X_test)

print("=== Nível Básico: Gaussian Naïve Bayes ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))