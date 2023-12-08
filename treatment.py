import os
import random
import shutil


# Definir diretórios de origem e destino
diretorio_origem = 'D:/DATASET/'

diretorio_destino_treino = 'D:/DATASET/dataset_treino'
diretorio_destino_teste = 'D:/DATASET/dataset_teste'


# Criar diretórios de destino se não existirem
if not os.path.exists(diretorio_destino_treino):
    os.makedirs(diretorio_destino_treino)

if not os.path.exists(diretorio_destino_teste):
    os.makedirs(diretorio_destino_teste)

# Lista das classes
classes = ['MATURE', 'IMMATURE']

# Iterar sobre as classes
for classe in classes:
    diretorio_origem_classe = os.path.join(diretorio_origem, classe)
    diretorio_destino_treino_classe = os.path.join(diretorio_destino_treino, classe)
    diretorio_destino_teste_classe = os.path.join(diretorio_destino_teste, classe)

    # Criar diretórios de destino para a classe se não existirem
    if not os.path.exists(diretorio_destino_treino_classe):
        os.makedirs(diretorio_destino_treino_classe)

    if not os.path.exists(diretorio_destino_teste_classe):
        os.makedirs(diretorio_destino_teste_classe)

    # Obter lista de imagens da classe
    imagens = os.listdir(diretorio_origem_classe)

    # Calcular quantidade de imagens para treinamento e teste
    qtd_treino = int(len(imagens) * 0.8)

    # Selecionar imagens de treinamento aleatoriamente
    imagens_treino = random.sample(imagens, qtd_treino)

    # Mover imagens de treinamento para o diretório de destino
    for imagem in imagens_treino:
        origem = os.path.join(diretorio_origem_classe, imagem)
        destino = os.path.join(diretorio_destino_treino_classe, imagem)
        shutil.copy(origem, destino)

    # Mover imagens restantes para o diretório de teste
    for imagem in imagens:
        if imagem not in imagens_treino:
            origem = os.path.join(diretorio_origem_classe, imagem)
            destino = os.path.join(diretorio_destino_teste_classe, imagem)
            shutil.copy(origem, destino)

print("Conjuntos de treinamento e teste criados com sucesso!")