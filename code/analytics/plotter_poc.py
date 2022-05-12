from matplotlib import pyplot as plt
from math import ceil

from data_provider.models import Sample
from data_provider.dataset import CustomDataset


def plot_samples(samples: list[Sample]):
    plt.rcParams['figure.figsize'] = (5, 5)
    col1 = 10
    row1 = ceil(len(samples)/col1)
    fig = plt.figure(figsize=(col1, row1))
    for index in range(0, col1 * row1):
        sample = samples[index]
        fig.add_subplot(row1, col1, index + 1)
        plt.axis('off')
        plt.imshow(sample.value)
        plt.title(f'{sample.class_label.name}', fontdict={'fontsize': 8, 'fontweight': 'medium'})
    plt.tight_layout()
    plt.show()


def plot_client_data_distribution(dataset: CustomDataset):
    plt.rcParams['figure.figsize'] = (18, 9)
    class_labels = [sample.name for sample in dataset.number_of_samples_per_class.keys()]
    class_labels_numbers = list(dataset.number_of_samples_per_class.values())
    plt.bar(class_labels, class_labels_numbers, color='maroon')
    plt.ylabel('Number of samples')
    plt.xticks(rotation=90)
    plt.yticks(fontsize=8)
    plt.show()


def plot_accuracy_comparison(acc, val_acc):
    """
    Plots accuracy comparison between two given.
    """
    plt.rcParams['figure.figsize'] = (25.0, 5.0) # set default size of plots
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Comparison of training and validation accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()


def plot_loss_comparison(loss, val_loss):
    """
    Plots loss comparison between two given.
    """
    plt.rcParams['figure.figsize'] = (25.0, 5.0) # set default size of plots
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Comparison of training and validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()
