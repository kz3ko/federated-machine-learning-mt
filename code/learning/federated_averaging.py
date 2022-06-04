from numpy import array, average


def count_clients_models_averaged_weights(clients_models_weights: list[list[array]]) -> list[array]:
    number_of_clients = len(clients_models_weights)
    relative_weights = [1 / number_of_clients for _ in range(number_of_clients)]
    clients_models_averaged_weights = []
    for weights_list_tuple in zip(*clients_models_weights):
        averaged_weights = array(
            [average(array(weights), axis=0, weights=relative_weights) for weights in zip(*weights_list_tuple)]
        )
        clients_models_averaged_weights.append(averaged_weights)

    return clients_models_averaged_weights
