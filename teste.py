from collections import defaultdict
from PIL import Image
import zipfile
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

'''
# Caminho para o arquivo enviado
uploaded_file_path = 'archive.zip'

extracted_dir = 'extracted_images/'

# Extraindo o conteúdo do arquivo
with zipfile.ZipFile(uploaded_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
'''


# Caminho para o arquivo enviado
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

# Verificar as categorias
for file_path in extracted_files:
    # Extraindo a categoria do caminho (assumindo que está no formato .../categoria/nome_imagem.jpg)
    category = os.path.basename(os.path.dirname(file_path))
    categories[category] += 1

# Listar as categorias e suas contagens
categories_sorted = sorted(categories.items(), key=lambda x: x[0])
print(categories_sorted)

# Definir tamanho fixo para as imagens
image_size = (64, 64)

# Listas para armazenar os dados e os rótulos
X = []
y = []

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

print(f"Dimensões de X: {X.shape}, Dimensões de y: {y.shape}")  # Confirmar dimensões

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

