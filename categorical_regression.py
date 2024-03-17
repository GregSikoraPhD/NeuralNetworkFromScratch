import nnfs
from nnfs.datasets import spiral_data

from activation_functions import Activation_ReLU, Activation_Softmax
from accuracy_measures import Accuracy_Categorical
from layers import Layer_Dense, Layer_Dropout
from loss_functions import Loss_CategoricalCrossentropy
from model import Model
from optimizers import Optimizer_Adam

nnfs.init()

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, accuracy
model.set(loss=Loss_CategoricalCrossentropy(),
          optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
          accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=1000, print_every=100)
model.plot_training()