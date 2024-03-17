import nnfs
from nnfs.datasets import sine_data

from activation_functions import Activation_Linear, Activation_ReLU
from accuracy_measures import Accuracy_Regression
from layers import Layer_Dense
from loss_functions import Loss_MeanSquaredError
from model import Model
from optimizers import Optimizer_Adam

nnfs.init()

# Create dataset
X, y = sine_data()

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss, optimizer, accuracy
model.set(loss=Loss_MeanSquaredError(),
          optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
          accuracy=Accuracy_Regression())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=500, print_every=100)
model.plot_training()
