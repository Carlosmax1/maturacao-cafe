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

def initialize_model(model_name, num_classes):
    model = None
    input_size = 0

    if model_name == "alexnet":
        model = models.alexnet(weights='DEFAULT')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model = models.vgg16(weights='DEFAULT')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size


def train_model(model, dataloaders, optimizer, basic_parameters, fold, date_now, device='gpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    model_name = basic_parameters.get('model_name')
    num_epochs = basic_parameters.get('epochs')
    batch_size = basic_parameters.get('batch_size')

    output_dir = r'outputs/' + model_name
    os.makedirs(output_dir, exist_ok=True)

    result_dir = output_dir + '\\' + model_name + '_' + date_now
    os.makedirs(result_dir, exist_ok=True)

    f = open(f'{result_dir}/{model_name}_fold_{fold}.txt', 'w')

    for epoch in range(num_epochs):
        f.write(f'Epoch {epoch}/{num_epochs - 1}\n')
        f.write('-' * 10 + '\n')

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            time_epoch_start = time.time()

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                model.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = basic_parameters.get('criterion')(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_epoch = time.time() - time_epoch_start

            f.write(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ({time_epoch:.4f} seconds) \n')

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ({time_epoch:.4f} seconds)')

            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_epoch = time.time() - since

        f.write(f'Time: {time_epoch:.0f}s\n')
        f.write('\n')

        print(f'Time: {time_epoch:.0f}s')
        print('\n')

    time_elapsed = time.time() - since
    f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    f.write(f'Number of epochs: {num_epochs}. Batch size: {batch_size}\n')
    f.write(f'Best val loss: {best_loss:.4f} Best val acc: {best_acc:.4f}\n')

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f} Best val acc: {best_acc:.4f}')

    y_true, y_pred = evaluate_model(model, dataloaders['val'], device=device)
    conf_mat_val = metrics.confusion_matrix(y_true, y_pred)
    f.write(f'\nConfusion Matrix:\n{conf_mat_val}\n')

    class_rep_val = generate_classification_report(model, dataloaders['val'],
                                                   basic_parameters.get('class_names'), device)
    f.write(f'\nClassification report:\n{class_rep_val}\n')

    f.close()

    plt.figure()
    plot_confusion_matrix(conf_mat_val, classes=basic_parameters.get('class_names'))
    plt.savefig(f'{result_dir}/{model_name}_fold_{fold}_cf_mat.pdf')

    plot_loss_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, model_name, fold, result_dir)

    model.load_state_dict(best_model_wts)
    return model



def evaluate_model(model, dataloader, device):
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    return y_true, y_pred

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Class')
    plt.xlabel('Predicted Class')

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, model_name, fold, save_dir):
    epochs = len(train_losses)
    x = range(1, epochs + 1)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, c='red', ls='-', label='Train loss', fillstyle='none')
    plt.plot(x, val_losses, c='blue', ls='--', label='Val. loss', fillstyle='none')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, [acc.cpu() for acc in train_accs], c='red', ls='-', label='Train accuracy', fillstyle='none')
    plt.plot(x, [acc.cpu() for acc in val_accs], c='blue', ls='--', label='Val. accuracy', fillstyle='none')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig(f'{save_dir}/{model_name}_fold_{fold}_loss_acc.pdf')

def generate_classification_report(model, dataloader, class_names, device='cpu'):
    model = model.to(device)
    model.eval()

    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    report = metrics.classification_report(
        all_labels.cpu().numpy(), all_preds.cpu().numpy(),
        target_names=class_names, digits=4, zero_division=0
    )

    return report


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

train_dir = "C:/DATASET/dataset_treino"
test_dir = "C:/DATASET/dataset_teste"

dataset = ImageFolder(test_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('\nDevice: {0}'.format(device))
    print(torch.cuda.get_device_name(0))

    for model in ['alexnet', 'vgg']:
        print("***********************************************************************")
        print(model)
        basic_parameters = {
            'num_classes': len(dataset.classes),
            'class_names': dataset.classes,
            'batch_size': 32,
            'lr': 0.001,
            'mm': 0.9,
            'epochs': 15,
            'model_name': model,
            'criterion': nn.CrossEntropyLoss()
        }

        model_ft, input_size = initialize_model(basic_parameters.get('model_name'), basic_parameters.get('num_classes'))

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([input_size, input_size], antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([input_size, input_size], antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
        test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=basic_parameters.get('batch_size'),
                                                   shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=basic_parameters.get('batch_size'),
                                                 shuffle=True, num_workers=4)

        dataloaders_dict = {'train': train_loader, 'val': val_loader}

        optimizer = optim.SGD(model_ft.parameters(), lr=basic_parameters.get('lr'), momentum=basic_parameters.get('mm'))

        date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_ft = train_model(model_ft, dataloaders_dict, optimizer, basic_parameters, 1, date_now, device)
