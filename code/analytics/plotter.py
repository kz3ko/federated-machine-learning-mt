from data_provider.dataset import CustomDataset


class ConfusionMatrix:

    def __init__(self, dataset: CustomDataset):
        self.dataset = dataset
