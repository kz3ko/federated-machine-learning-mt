from abc import ABC, abstractmethod

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
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
            restore_best_weights=early_stopping_config.restore_best_weights
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

    """
    Simplest model used for training.
    =================================================================
    Total params: 258,762
    Trainable params: 258,762
    Non-trainable params: 0
    =================================================================
    Epoch learning time: 14-18s
    Epochs until "EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)" stops training:
    Stats in best - 16 - epoch:
        - accuracy - 0.8092
        - loss - 0.5521
        - val accuracy - 0.6960
        - val loss - 0.9122
        - test_accuracy - 0.7006
        - test_loss - 0.9133
    """

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


class SecondNeuralNetworkModel(FirstNeuralNetworkModel):

    """
    "FirstNeuralNetworkModel" with BatchNormalization layers after Conv2D layers.
    =================================================================
    Total params: 259,658
    Trainable params: 259,210
    Non-trainable params: 448
    =================================================================
    Epoch learning time: 27-32s
    Epochs until "EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)" stops training:
    Stats in best - 4 - epoch:
        - accuracy - 0.7541
        - loss - 0.7045
        - val accuracy - 0.6701
        - val loss - 0.9502
        - test_accuracy - 0.6635
        - test_loss - 0.9559
    """

    @staticmethod
    def _create_base_model() -> Sequential:
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        model.summary()

        return model


class ThirdNeuralNetworkModel(FirstNeuralNetworkModel):

    """
    "SecondNeuralNetworkModel" with Droput layers after MaxPooling layers.
    =================================================================
    Total params: 259,658
    Trainable params: 259,210
    Non-trainable params: 448
    =================================================================
    Epoch learning time: 27-43s
    Epochs until "EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)" stops training:
    Stats in best - 13 - epoch:
        - accuracy - 0.8097
        - loss - 0.5410
        - val accuracy - 0.7399
        - val loss - 0.7956
        - test_accuracy - 0.7294
        - test_loss - 0.8218
    """

    @staticmethod
    def _create_base_model() -> Sequential:
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        model.summary()

        return model


class FourthNeuralNetworkModel(FirstNeuralNetworkModel):

    """
    "ThirdNeuralNetworkModel" with l1 kernel regularizers in Dense layers.
    =================================================================
    Total params: 259,658
    Trainable params: 259,210
    Non-trainable params: 448
    =================================================================
    Epoch learning time: 28-35s
    Epochs until "EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)" stops training:
    Stats in best - 22 - epoch:
        - accuracy - 0.7935
        - loss - 0.7423
        - val accuracy - 0.7708
        - val loss - 0.8264
        - test_accuracy - 0.7704
        - test_loss - 0.8235
    """

    @staticmethod
    def _create_base_model() -> Sequential:
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=l1(0.001)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l1(0.001)))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        model.summary()

        return model
