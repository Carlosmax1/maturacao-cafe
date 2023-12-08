import os
import random
import time
import datetime
import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

from efficientnet_pytorch import EfficientNet

import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def inicializar_modelo(nome_modelo, num_classes):
    modelo = None
    tamanho_entrada = 0

    if nome_modelo == "alexnet":
        modelo = models.alexnet(weights='DEFAULT')
        num_ftrs = modelo.classifier[6].in_features
        modelo.classifier[6] = nn.Linear(num_ftrs, num_classes)
        tamanho_entrada = 224

    elif nome_modelo == "vgg":
        modelo = models.vgg16(weights='DEFAULT')
        num_ftrs = modelo.classifier[6].in_features
        modelo.classifier[6] = nn.Linear(num_ftrs, num_classes)
        tamanho_entrada = 224

    else:
        print("Nome de modelo inválido, saindo...")
        exit()

    return modelo, tamanho_entrada

def treinar_modelo(modelo, dataloaders, otimizador, parametros_basicos, fold, data_atual, dispositivo='gpu'):
    desde = time.time()

    melhores_pesos_modelo = copy.deepcopy(modelo.state_dict())
    melhor_acuracia = 0.0
    melhor_perda = float('inf')

    # Listas para armazenar perdas e acurácias
    perda_treino_lista = []
    acuracia_treino_lista = []
    perda_val_lista = []
    acuracia_val_lista = []

    nome_modelo = parametros_basicos.get('model_name')
    num_epochs = parametros_basicos.get('epochs')
    tamanho_batch = parametros_basicos.get('batch_size')

    diretorio_saida = r'outputs/' + nome_modelo
    os.makedirs(diretorio_saida, exist_ok=True)

    diretorio_resultado = diretorio_saida + '\\' + nome_modelo + '_' + data_atual
    os.makedirs(diretorio_resultado, exist_ok=True)

    arquivo = open(f'{diretorio_resultado}/{nome_modelo}_fold_{fold}.txt', 'w')

    for epoca in range(num_epochs):
        arquivo.write(f'Época {epoca}/{num_epochs - 1}\n')
        arquivo.write('-' * 10 + '\n')

        print(f'Época {epoca}/{num_epochs - 1}')
        print('-' * 10)

        for fase in ['treino', 'val']:
            tempo_epoca_inicio = time.time()

            if fase == 'treino':
                modelo.train()
            else:
                modelo.eval()

            perda_atual = 0.0
            acertos_atual = 0

            for entradas, rotulos in dataloaders[fase]:
                entradas = entradas.to(dispositivo)
                rotulos = rotulos.to(dispositivo)

                modelo.to(dispositivo)

                otimizador.zero_grad()

                with torch.set_grad_enabled(fase == 'treino'):
                    saidas = modelo(entradas)
                    perda = parametros_basicos.get('criterion')(saidas, rotulos)

                    _, previsoes = torch.max(saidas, 1)

                    if fase == 'treino':
                        perda.backward()
                        otimizador.step()

                perda_atual += perda.item() * entradas.size(0)
                acertos_atual += torch.sum(previsoes == rotulos.data)

            perda_epoca = perda_atual / len(dataloaders[fase].dataset)
            acuracia_epoca = acertos_atual.double() / len(dataloaders[fase].dataset)

            tempo_epoca = time.time() - tempo_epoca_inicio

            arquivo.write(f'{fase.capitalize()} Perda: {perda_epoca:.4f} Acurácia: {acuracia_epoca:.4f} ({tempo_epoca:.4f} segundos) \n')

            print(f'{fase.capitalize()} Perda: {perda_epoca:.4f} Acurácia: {acuracia_epoca:.4f} ({tempo_epoca:.4f} segundos)')

            if fase == 'treino':
                perda_treino_lista.append(perda_epoca)
                acuracia_treino_lista.append(acuracia_epoca)
            else:
                perda_val_lista.append(perda_epoca)
                acuracia_val_lista.append(acuracia_epoca)

            if fase == 'val' and perda_epoca < melhor_perda:
                melhor_perda = perda_epoca
                melhor_acuracia = acuracia_epoca
                melhores_pesos_modelo = copy.deepcopy(modelo.state_dict())

        tempo_epoca = time.time() - desde

        arquivo.write(f'Tempo: {tempo_epoca:.0f}s\n')
        arquivo.write('\n')

        print(f'Tempo: {tempo_epoca:.0f}s')
        print('\n')

    tempo_total = time.time() - desde
    arquivo.write(f'Treinamento completo em {tempo_total // 60:.0f}m {tempo_total % 60:.0f}s\n')
    arquivo.write(f'Número de épocas: {num_epochs}. Tamanho do batch: {tamanho_batch}\n')
    arquivo.write(f'Melhor perda de validação: {melhor_perda:.4f} Melhor acurácia de validação: {melhor_acuracia:.4f}\n')

    print(f'Treinamento completo em {tempo_total // 60:.0f}m {tempo_total % 60:.0f}s')
    print(f'Melhor perda de validação: {melhor_perda:.4f} Melhor acurácia de validação: {melhor_acuracia:.4f}')

    y_true, y_pred = avaliar_modelo(modelo, dataloaders['val'], dispositivo=dispositivo)
    matriz_confusao_val = metrics.confusion_matrix(y_true, y_pred)
    arquivo.write(f'\nMatriz de Confusão:\n{matriz_confusao_val}\n')

    relatorio_classes_val = gerar_relatorio_classificacao(modelo, dataloaders['val'],
                                                   parametros_basicos.get('class_names'), dispositivo)
    arquivo.write(f'\nRelatório de Classificação:\n{relatorio_classes_val}\n')

    arquivo.close()

    plt.figure()
    plotar_matriz_confusao(matriz_confusao_val, classes=parametros_basicos.get('class_names'))
    plt.savefig(f'{diretorio_resultado}/{nome_modelo}_fold_{fold}_matriz_confusao.pdf')

    plotar_perda_acuracia(perda_treino_lista, perda_val_lista, acuracia_treino_lista, acuracia_val_lista, nome_modelo, fold, diretorio_resultado)

    modelo.load_state_dict(melhores_pesos_modelo)
    return modelo

def avaliar_modelo(modelo, dataloader, dispositivo):
    y_true = []
    y_pred = []
    corretas = 0
    total = 0
    modelo.eval()

    with torch.no_grad():
        for entradas, rotulos in dataloader:
            entradas, rotulos = entradas.to(dispositivo), rotulos.to(dispositivo)
            saidas = modelo(entradas)
            _, previsto = torch.max(saidas.data, 1)
            total += rotulos.size(0)
            corretas += (previsto == rotulos).sum().item()
            y_true += rotulos.tolist()
            y_pred += previsto.tolist()

    return y_true, y_pred

def plotar_matriz_confusao(cm, classes, titulo='Matriz de Confusão', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(titulo)
    plt.colorbar()
    marcas_eixo_x = np.arange(len(classes))
    plt.xticks(marcas_eixo_x, classes, rotation=45)
    plt.yticks(marcas_eixo_x, classes)

    limite = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > limite else "black")

    plt.tight_layout()
    plt.ylabel('Classe')
    plt.xlabel('Classe Prevista')
    plt.show()

def plotar_perda_acuracia(perda_treino, perda_val, acuracia_treino, acuracia_val, nome_modelo, fold, diretorio_salvar):
    epocas = len(perda_treino)
    x = range(1, epocas + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, perda_treino, c='red', ls='-', label='Perda no treino', fillstyle='none')
    plt.plot(x, perda_val, c='blue', ls='--', label='Perda na val.', fillstyle='none')
    plt.title('Perda')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, [acc.cpu() for acc in acuracia_treino], c='red', ls='-', label='Acurácia no treino', fillstyle='none')
    plt.plot(x, [acc.cpu() for acc in acuracia_val], c='blue', ls='--', label='Acurácia na val.', fillstyle='none')
    plt.title('Acurácia')
    plt.legend()

    plt.savefig(f'{diretorio_salvar}/{nome_modelo}_fold_{fold}_perda_acuracia.pdf')
    print(f'{nome_modelo}')
    plt.show()

def gerar_relatorio_classificacao(modelo, dataloader, nomes_classes, dispositivo='cpu'):
    modelo = modelo.to(dispositivo)
    modelo.eval()

    todas_predicoes = torch.tensor([], dtype=torch.long, device=dispositivo)
    todos_rotulos = torch.tensor([], dtype=torch.long, device=dispositivo)

    for entradas, rotulos in dataloader:
        entradas = entradas.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        with torch.no_grad():
            saidas = modelo(entradas)
            _, predicoes = torch.max(saidas, 1)

        todas_predicoes = torch.cat((todas_predicoes, predicoes), dim=0)
        todos_rotulos = torch.cat((todos_rotulos, rotulos), dim=0)

    relatorio = metrics.classification_report(
        todos_rotulos.cpu().numpy(), todas_predicoes.cpu().numpy(),
        target_names=nomes_classes, digits=4, zero_division=0
    )

    return relatorio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

diretorio_treino = "DATASET/dataset_treino"
diretorio_teste = "DATASET/dataset_teste"

conjunto_dados = ImageFolder(diretorio_teste)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('\nDispositivo: {0}'.format(device))
    #print(torch.cuda.get_device_name(0))

    for modelo in ['alexnet','vgg']:
        print("***********************************************************************")
        print(modelo)
        parametros_basicos = {
            'num_classes': len(conjunto_dados.classes),
            'class_names': conjunto_dados.classes,
            'batch_size': 32,
            'lr': 0.001,
            'mm': 0.9,
            'epochs': 15,
            'model_name': modelo,
            'criterion': nn.CrossEntropyLoss()
        }

        modelo_ft, tamanho_entrada = inicializar_modelo(parametros_basicos.get('model_name'), parametros_basicos.get('num_classes'))

        transformacoes_dados = {
            'treino': transforms.Compose([
                transforms.Resize([tamanho_entrada, tamanho_entrada], antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([tamanho_entrada, tamanho_entrada], antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        conjunto_treino = datasets.ImageFolder(diretorio_treino, transform=transformacoes_dados['treino'])
        conjunto_teste = datasets.ImageFolder(diretorio_teste, transform=transformacoes_dados['val'])

        carregador_treino = torch.utils.data.DataLoader(conjunto_treino, batch_size=parametros_basicos.get('batch_size'),
                                                        shuffle=True, num_workers=4)
        carregador_val = torch.utils.data.DataLoader(conjunto_teste, batch_size=parametros_basicos.get('batch_size'),
                                                     shuffle=True, num_workers=4)

        dicionario_dataloaders = {'treino': carregador_treino, 'val': carregador_val}

        otimizador = optim.SGD(modelo_ft.parameters(), lr=parametros_basicos.get('lr'), momentum=parametros_basicos.get('mm'))

        data_atual = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        modelo_ft = treinar_modelo(modelo_ft, dicionario_dataloaders, otimizador, parametros_basicos, 1, data_atual, device)