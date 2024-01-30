import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

from neural_network import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_Adam
# ===============================Creating objects=====================================
# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation
activation1 = Activation_ReLU()

# Create second Dense layer
dense2 = Layer_Dense(64, 3)

# Create softmax classifier combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
# optimizer = Optimizer_SGD(decay=0.001, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-4, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)
# ====================================Training===========================================
for epoch in range(10001):
    # ===================================Forward pass ===================================
    # Perform a forward pass of the first layer
    dense1.forward(X)

    # Perform a forward pass of activation ReLU
    activation1.forward(dense1.output)

    # Perform a forward pass of second layer
    dense2.forward(activation1.output)

    # Data loss
    data_loss = loss_activation.forward(dense2.output, y)

    # Regularization loss
    regularization_loss = (loss_activation.loss.regularization_loss(dense1) +
                           loss_activation.loss.regularization_loss(dense2))

    # Overall loss
    loss = data_loss + regularization_loss
    # ========================Accuracy =================================================
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    # ====================Printing info about training==================================
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}, )' +
              f'lr: {optimizer.current_learning_rate}')
    # ======================== Backward pass ===========================================
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # ============================Update weights and biases========================================
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()