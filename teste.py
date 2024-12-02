from collections import defaultdict
from PIL import Image
import zipfile
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt



'''
# Caminho para o arquivo enviado
uploaded_file_path = 'archive.zip'

extracted_dir = 'extracted_images/'

# Extraindo o conteúdo do arquivo
with zipfile.ZipFile(uploaded_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
'''


# Configuração inicial: Extração de dados
uploaded_file_path = 'archive - Copia (1).zip'
extracted_dir = 'extracted_images_teste/'

# Extraindo o conteúdo do arquivo
with zipfile.ZipFile(uploaded_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Listar os arquivos extraídos
extracted_files = []
for root, dirs, files in os.walk(extracted_dir):
    for file in files:
        extracted_files.append(os.path.join(root, file))

# Dicionário para armazenar a contagem de arquivos por categoria
categories = defaultdict(int)
for file_path in extracted_files:
    category = os.path.basename(os.path.dirname(file_path))
    categories[category] += 1

# Listar as categorias e suas contagens
categories_sorted = sorted(categories.items(), key=lambda x: x[0])
print("Categorias encontradas:", categories_sorted)

# Processamento das imagens
image_size = (64, 64)
X, y = [], []

# Processar as imagens
valid_images = []
valid_labels = []

# Processar as imagens
for file_path in extracted_files:
    try:
        # Carregar imagem
        img = Image.open(file_path)

        # Verificar o modo da imagem e converter para RGB ou RGBA
        if img.mode in ('P', 'RGBA'):  # Imagens com paleta ou transparência
            img = img.convert('RGBA')  # Converte para RGBA
        else:
            img = img.convert('RGB')  # Converte para RGB para outras imagens

        # Redimensionar para o tamanho fixo
        img_resized = img.resize(image_size)

        # Converter para array numpy
        img_array = np.array(img_resized) / 255.0

        # Ajustar dimensões (forçando o formato (64, 64, 3))
        if img_array.shape != (64, 64, 3):
            img_array = np.resize(img_array, (64, 64, 3))

        # Flatten (vetorizar a imagem) e adicionar aos dados válidos
        valid_images.append(img_array.flatten())

        # Extrair o rótulo da categoria
        category = os.path.basename(os.path.dirname(file_path))
        valid_labels.append(category)
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")


# Converter para arrays numpy
X = np.array(valid_images)
y = np.array(valid_labels)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configuração do SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # Kernel linear e margem padrão

# Treinamento do SVM
svm_model.fit(X_train, y_train)
print("Modelo SVM treinado com sucesso!")

# Previsões no conjunto de teste
y_pred = svm_model.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

report = classification_report(y_test, y_pred, target_names=np.unique(y))
print("Relatório de Classificação:")
print(report)

# Exibir vetores de suporte
support_vectors = svm_model.support_vectors_
print(f"Quantidade de vetores de suporte: {len(support_vectors)}")

# Visualização do hiperplano para dados bidimensionais (se aplicável)
if X_train.shape[1] == 2:
    w = svm_model.coef_
    b = svm_model.intercept_
    print("Coeficientes do hiperplano (w):", w)
    print("Intercepto do hiperplano (b):", b)

    # Gerar valores para x1
    x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    x2 = -(w[0, 0] * x1 + b[0]) / w[0, 1]

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label='Dados de Treino')
    plt.plot(x1, x2, color='black', label='Hiperplano')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, edgecolors='yellow', facecolors='none', label='Vetores de Suporte')
    plt.legend()
    plt.title("Hiperplano do SVM e Dados de Treinamento")
    plt.show()
else:
    print("Os dados possuem mais de 2 dimensões. Visualização do hiperplano não aplicável.")

# Função para treinar e avaliar com diferentes kernels
def test_kernels(X_train, X_test, y_train, y_test, kernels):
    results = {}
    for kernel in kernels:
        print(f"\nTestando com kernel: {kernel}")
        
        # Configuração do SVM com o kernel atual
        if kernel == 'poly':
            # Adicionando um grau ao kernel polinomial
            model = SVC(kernel=kernel, degree=3, C=1.0, random_state=42)
        else:
            model = SVC(kernel=kernel, C=1.0, random_state=42)

        # Treinamento do modelo
        model.fit(X_train, y_train)
        
        # Previsões e avaliação
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia com kernel '{kernel}': {accuracy:.2f}")
        
        # Salvar resultados para comparação
        results[kernel] = {
            "model": model,
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, target_names=np.unique(y))
        }

        # Exibir relatório de classificação
        print(f"Relatório para kernel '{kernel}':\n")
        print(results[kernel]["classification_report"])

    return results

# Lista de kernels para testar
kernels_to_test = ['linear', 'poly', 'rbf', 'sigmoid']

# Executar os testes
kernel_results = test_kernels(X_train, X_test, y_train, y_test, kernels_to_test)

# Comparar os resultados
print("\nComparação de Acurácias:")
for kernel, result in kernel_results.items():
    print(f"{kernel}: {result['accuracy']:.2f}")

from sklearn.model_selection import GridSearchCV

# Definir os hiperparâmetros a serem ajustados
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],  # Testar diferentes kernels
    'C': [0.1, 1, 10, 100],              # Valores de margem
    'gamma': ['scale', 0.1, 1, 10],      # Gamma para rbf e poly
    'degree': [2, 3, 4]                  # Graus do polinômio para kernel poly
}

# Instanciar o modelo SVM
svm_model = SVC(random_state=42)

# Configurar o GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Executar o ajuste
grid_search.fit(X_train, y_train)

# Exibir os melhores paryâmetros encontrados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# Exibir a acurácia obtida com os melhores parâmetros
print(f"Melhor acurácia obtida: {grid_search.best_score_:.2f}")

# Treinar o modelo com os melhores parâmetros no conjunto de treino completo
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Avaliar no conjunto de teste
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste com os melhores parâmetros: {accuracy:.2f}")

##teste