from tensorflow.keras.callbacks import History

from learning.participant import Client


class StatisticsCollector:

    def __init__(self, clients: list[Client]):
        self.client_metrics = {}
        self.server_metrics = {}

    def save_client_metrics(self, iteration: int, client: Client):
        new_metrics = self.__get_history_metrics(client.latest_learning_history)
        client_metrics = self.client_metrics.setdefault(client.id, {})
        client_metrics[iteration] = {}
        for name, value in new_metrics:
            client_metrics.setdefault(name, []).append(value)



    @staticmethod
    def __get_history_metrics(history: History) -> dict[str, float]:
        return {
            'accuracy': history.history['accuracy'],
            'validation_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'validation_loss': history.history['val_loss']
        }
