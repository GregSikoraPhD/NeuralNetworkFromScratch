import matplotlib.pyplot as plt
import time

from activation_functions import Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossentropy
from layers import Layer_Input
from loss_functions import Loss_CategoricalCrossentropy


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        start_time = time.time()

        # Initialize accuracy object
        self.accuracy.init(y)
        self.accuracy_vector = []
        self.loss_vector = []
        self.learning_rate_vector = []

        for epoch in range(1, epochs + 1):
            # Perform forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            self.loss_vector.append(loss)

            # Get predictions and compute accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            self.accuracy_vector.append(accuracy)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            self.learning_rate_vector.append(self.optimizer.current_learning_rate)

            # Print summary
            if not epoch % print_every:
                print(f'epoch: {epoch, }' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        # If there is validation data
        if validation_data is not None:
            X_val, y_val = validation_data

            # Perform the forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, y_val)

            # Get prediction and compute accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

        end_time = time.time()
        self.training_time = end_time - start_time

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # Case for Softmax activation and Categorical Cross-Entropy
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        # Call backward on the loss
        self.loss.backward(output, y)

        # Call backward for layers
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def plot_training(self):
        # Create a figure and axes for subplots
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Plotting each subplot
        axs[0].plot(self.accuracy_vector)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')

        axs[1].plot(self.loss_vector)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')

        axs[2].plot(self.learning_rate_vector)
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Learning rate')

        # Adding a common title for all subplots
        fig.suptitle(f'Training dynamics, training time = {self.training_time:.2f} s')

        # Adjust layout
        plt.tight_layout()

        # Display the plot
        plt.show()
