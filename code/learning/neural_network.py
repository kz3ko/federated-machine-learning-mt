from abc import ABC, abstractmethod

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.utils import to_categorical
from numpy import array, argmax

from data_provider.dataset import CustomDataset
from config.config import EarlyStoppingConfig
from learning.models import SingleTestMetrics, PredictionMetrics


class NeuralNetworkModel(ABC):

    history: History
    batch_size: int

    def __init__(self):
        self.base_model = self._create_base_model()
        self.epochs = 1
        self.callbacks = []

    def test(self, dataset: CustomDataset) -> SingleTestMetrics:
        loss, accuracy = self.base_model.evaluate(dataset.input_values, dataset.target_labels, self.batch_size)
        metrics = SingleTestMetrics(accuracy, loss)

        return metrics

    def make_predictions(self, dataset: CustomDataset, batch_size: int = 128) -> PredictionMetrics:
        labels = to_categorical(dataset.target_labels)
        steps = int(len(dataset)/batch_size)
        verbose = 1
        predicted_labels = self.base_model.predict(dataset.input_values, verbose=verbose, steps=steps)
        max_label = argmax(labels, axis=1)
        predicted_max_label = argmax(predicted_labels, axis=1)

        metrics = PredictionMetrics(labels, predicted_labels, max_label, predicted_max_label)

        return metrics

    def get_weights(self) -> list[array]:
        return self.base_model.get_weights()

    def set_weights(self, new_weights: list[array]):
        self.base_model.set_weights(new_weights)

    def save(self, target_path):
        self.base_model.save(target_path)

    def load(self, model_path):
        self.base_model = load_model(model_path)

    def update_early_stopping(self, early_stopping_config: EarlyStoppingConfig):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.patience,
            monitor=early_stopping_config.metric_type,
        )
        self.callbacks.append(early_stopping)

    @abstractmethod
    def train(self, dataset: CustomDataset) -> History:
        pass

    @staticmethod
    @abstractmethod
    def _create_base_model() -> Sequential:
        pass


class FirstNeuralNetworkModel(NeuralNetworkModel):

    def __init__(self):
        super().__init__()
        self.batch_size = 256
        self.verbosity = 1
        self.validation_split = 0.2

    def train(self, dataset: CustomDataset) -> History:
        self.history = self.base_model.fit(
            dataset.input_values,
            dataset.target_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbosity,
            validation_split=self.validation_split,
            shuffle=True,
            callbacks=self.callbacks
        )

        return self.history

    @staticmethod
    def _create_base_model() -> Sequential:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        model.summary()

        return model
