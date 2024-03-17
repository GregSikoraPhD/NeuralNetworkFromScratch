import nnfs
from nnfs.datasets import spiral_data

from activation_functions import Activation_ReLU, Activation_Sigmoid
from accuracy_measures import Accuracy_Categorical
from layers import Layer_Dense
from loss_functions import Loss_BinaryCrossentropy
from model import Model
from optimizers import Optimizer_Adam

nnfs.init()

# Create dataset
X, y = spiral_data(samples=1000, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())

# Set loss, optimizer, accuracy
model.set(loss=Loss_BinaryCrossentropy(),
          optimizer=Optimizer_Adam(decay=5e-7),
          accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=5000, print_every=100)
model.plot_training()
