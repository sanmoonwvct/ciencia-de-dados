import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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

# 5. Matriz de confusão
# aqui eu só monto a matriz e jogo num gráfico simples pra visualizar melhor
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de Confusão - Gaussian NB")
plt.show()

# 6. Curva ROC
# pego a probabilidade da classe positiva (coluna 1) pra calcular a curva
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # linha de referência (chute aleatório)
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - Gaussian NB")
plt.legend()
plt.show()

print(f"\nAUC da curva ROC: {roc_auc:.2f}")

# 7. Salvando o modelo treinado em .pkl
with open("modelo_gaussian_nb.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModelo salvo em: modelo_gaussian_nb.pkl")
