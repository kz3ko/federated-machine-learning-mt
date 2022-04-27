from abc import ABC, abstractmethod

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import History

from data_provider.dataset import ClientDataset


class NeuralNetworkModel(ABC):

    def __init__(self, dataset: ClientDataset):
        self.input_values = dataset.input_values
        self.target_labels = dataset.target_labels
        self.history = None
        self.base_model = self._create_base_model()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _create_base_model(self) -> Sequential:
        pass


class FirstNeuralNetworkModel(NeuralNetworkModel):

    def train(self) -> History:
        number_of_epochs = 50
        batch_size = 256
        verbosity = 1
        validation_split = 0.2

        self.history = self.base_model.fit(
            self.input_values,
            self.target_labels,
            batch_size=batch_size,
            epochs=number_of_epochs,
            verbose=verbosity,
            validation_split=validation_split,
            shuffle=True
        )

        return self.history

    def _create_base_model(self) -> Sequential:
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
        model.add(Dense(100, activation='softmax'))

        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        model.summary()

        return model
