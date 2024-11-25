import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        if activation == "tanh":
            self.activation_fn = self.tanh
        elif activation == "sigmoid":
            self.activation_fn = self.sigmoid
        else:
            self.activation_fn = self.softmax
        # TODO: define layers and initialize weights
        self.weights_input_hidden = np.random.randn(input_dim,hidden_dim)
        self.weights_hidden_output = np.random.randn(hidden_dim,output_dim)
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.bias_output = np.zeros((1, output_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation_fn(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.activation_fn(self.final_input)
        out = self.final_output
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule

        # TODO: update weights with gradient descent

        # TODO: store gradients for visualization

        output_error =  self.final_output - y

        # chain rule for tanh
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (1 - self.hidden_output ** 2)

        self.weights_hidden_output -= self.lr * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= self.lr * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= self.lr * np.dot(X.T, hidden_error)
        self.bias_hidden -= self.lr * np.sum(hidden_error, axis=0, keepdims=True)


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden_output
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f"Hidden Space at Step {10 * frame}")

        # Generate grid for two hidden dimensions
    h1_min, h1_max = hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1
    h2_min, h2_max = hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1
    hxx, hyy = np.meshgrid(np.linspace(h1_min, h1_max, 50), np.linspace(h2_min, h2_max, 50))
    hidden_grid = np.c_[hxx.ravel(), hyy.ravel(), np.zeros(hxx.size)]  # Assume third hidden feature is constant

    # Transform the grid through the second layer of the network
    hidden_grid_outputs = np.dot(
        hidden_grid[:, :2], mlp.weights_hidden_output[:2, :]
    ) + mlp.bias_output
    hidden_grid_outputs = hidden_grid_outputs.reshape(hxx.shape)

    # Plot the surface in the hidden space
    ax_hidden.plot_surface(
        hxx, hyy, hidden_grid_outputs, cmap='bwr', alpha=0.6, edgecolor='none'
    )

    # TODO: Hyperplane visualization in the hidden space

    # TODO: Distorted input space transformed by the hidden layer
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_outputs = mlp.forward(grid_points)  # Compute predictions for the grid
    grid_outputs = grid_outputs.reshape(xx.shape)  # Reshape for plotting
    ax_input.contourf(xx, yy, grid_outputs, levels=50, cmap='bwr', alpha=0.6)


    # TODO: Plot input layer decision boundary
    input_features = X
    ax_input.scatter(input_features[:, 0], input_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title(f"Input Space at Step {10 * frame}")

    # TODO: Visualize features and gradients as circles and edges 
    ax_gradient.set_title("Gradients at Step {}".format(10 * frame))
    # plot input node, hidden node and the output node
    ax_gradient.add_patch(Circle((0.2, 0.2), radius=0.02, color='b'))
    ax_gradient.text(0.2, 0.15, "x1", ha="center", va="top", fontsize=8)
    ax_gradient.add_patch(Circle((0.2, 0.8), radius=0.02, color='b'))
    ax_gradient.text(0.2, 0.85, "x2", ha="center", va="bottom", fontsize=8)
    ax_gradient.add_patch(Circle((0.8, 0.2), radius=0.02, color='b'))
    ax_gradient.text(0.85, 0.2, "y", ha="left", va="center", fontsize=8)
    ax_gradient.add_patch(Circle((0.5, 0.2), radius=0.02, color='b'))
    ax_gradient.text(0.5, 0.15, "h1", ha="center", va="top", fontsize=8)
    ax_gradient.add_patch(Circle((0.5, 0.5), radius=0.02, color='b'))
    ax_gradient.text(0.5, 0.45, "h2", ha="center", va="top", fontsize=8)
    ax_gradient.add_patch(Circle((0.5, 0.8), radius=0.02, color='b'))
    ax_gradient.text(0.5, 0.85, "h3", ha="center", va="bottom", fontsize=8)

    # The edge thickness visually represents the magnitude of the gradient
    ax_gradient.plot([0.2, 0.5], [0.2, 0.2], 'k-', lw=mlp.weights_input_hidden[0][0])  # x1 to h1
    ax_gradient.plot([0.2, 0.5], [0.2, 0.5], 'k-', lw=mlp.weights_input_hidden[0][1])  # x1 to h2
    ax_gradient.plot([0.2, 0.5], [0.2, 0.8], 'k-', lw=mlp.weights_input_hidden[0][2])  # x1 to h3
    ax_gradient.plot([0.2, 0.5], [0.8, 0.2], 'k-', lw=mlp.weights_input_hidden[1][0])  # x2 to h1
    ax_gradient.plot([0.2, 0.5], [0.8, 0.5], 'k-', lw=mlp.weights_input_hidden[1][1])  # x2 to h2
    ax_gradient.plot([0.2, 0.5], [0.8, 0.8], 'k-', lw=mlp.weights_input_hidden[1][2])  # x2 to h3

    # From hidden layer (h1, h2, h3) to output (y)
    ax_gradient.plot([0.5, 0.8], [0.2, 0.2], 'k-', lw=mlp.weights_hidden_output[0])  # h1 to y
    ax_gradient.plot([0.5, 0.8], [0.5, 0.2], 'k-', lw=mlp.weights_hidden_output[1])  # h2 to y
    ax_gradient.plot([0.5, 0.8], [0.8, 0.2], 'k-', lw=mlp.weights_hidden_output[2])  # h3 to y

    

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.01
    step_num = 1000
    visualize(activation, lr, step_num)