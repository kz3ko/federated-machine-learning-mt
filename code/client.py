from dataset import ClientDataset


class Client:

    def __init__(self, client_id: int, dataset: ClientDataset):
        self.id = client_id
        self.dataset = dataset
