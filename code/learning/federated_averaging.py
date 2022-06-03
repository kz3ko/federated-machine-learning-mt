from numpy import array, average

from learning.participant import Client


class FederatedAveraging:

    def __init__(self, clients: list[Client]):
        self.clients = clients

    def get_clients_models_averaged_weights(self, clients_models_weights: list[list[array]]) -> list[array]:
        relative_weights = [1 / len(self.clients) for _ in self.clients]
        clients_models_averaged_weights = []
        for weights_list_tuple in zip(*clients_models_weights):
            averaged_weights = array(
                [average(array(weights), axis=0, weights=relative_weights) for weights in zip(*weights_list_tuple)]
            )
            clients_models_averaged_weights.append(averaged_weights)

        return clients_models_averaged_weights
