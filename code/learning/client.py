from learning.neural_network import NeuralNetworkModel
from data_provider.dataset import ClientDataset


class Client:

    def __init__(self, client_id: int, dataset: ClientDataset, model: NeuralNetworkModel):
        self.id = client_id
        self.dataset = dataset
        self.model = model
        self.history = None

    def train_model(self) -> NeuralNetworkModel:
        self.history = self.model.train()

        return self.model
