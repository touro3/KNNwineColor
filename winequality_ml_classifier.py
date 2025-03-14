import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# =============================================================================
# 1. Carregar e Preparar os Dados
# =============================================================================
data_path = 'winequality-merged.csv'  # Ajuste o caminho conforme necessário
wine_data = pd.read_csv(data_path)

# Converter a coluna 'color' para numérico: 0 = Red, 1 = White
wine_data['color'] = wine_data['color'].map({'red': 0, 'white': 1})

# Exibir a distribuição original das classes
print("Distribuição original das classes:", Counter(wine_data['color']))

# =============================================================================
# 2. Balanceamento de Classes com SMOTE
# =============================================================================
X = wine_data.drop(columns=['color'])  # Todas as features, exceto 'color'
y = wine_data['color']

smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Distribuição após SMOTE:", Counter(y_resampled))

# =============================================================================
# 3. Divisão dos Dados e Padronização
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42, stratify=y_resampled)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =============================================================================
# 4. Treinamento do Modelo KNN com Parâmetros Otimizados (via GridSearch)
# =============================================================================
# Parâmetros otimizados encontrados:
# {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'manhattan', 'n_neighbors': 10, 'p': 1, 'weights': 'distance'}
knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='distance', leaf_size=10, p=1)
knn.fit(X_train_scaled, y_train)

# =============================================================================
# 5. Avaliação do Modelo
# =============================================================================
# Previsões e métricas
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\n📌 Acurácia do modelo KNN após SMOTE:", accuracy)
print("\n📌 Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=['Red', 'White']))
print("📊 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# =============================================================================
# 6. Plotagem da Curva ROC
# =============================================================================
# Obter as probabilidades preditas para a classe positiva (1 - White)
y_probs = knn.predict_proba(X_test_scaled)[:, 1]

# Calcular FPR, TPR e limiares
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
