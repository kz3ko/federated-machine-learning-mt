from matplotlib import pyplot as plt
from math import ceil

from data_provider.models import Sample


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
