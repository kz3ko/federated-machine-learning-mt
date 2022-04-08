from matplotlib import pyplot as plt
from math import ceil

from data_provider.models import Sample
from data_provider.dataset import ClientDataset


def plot_samples(samples: list[Sample]):
    plt.rcParams['figure.figsize'] = (5, 5)
    col1 = 5
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


def plot_client_data_distribution(dataset: ClientDataset):
    plt.rcParams['figure.figsize'] = (5, 12)
    class_labels = [sample.name for sample in dataset.number_of_samples_per_class.keys()]
    class_labels_numbers = list(dataset.number_of_samples_per_class.values())
    plt.barh(class_labels, class_labels_numbers, color='maroon')
    plt.yticks(fontsize=8)
    plt.show()
