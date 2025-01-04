import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

g_seed: int = 52

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(num_samples, num_inputs):
    X = np.random.rand(num_samples, num_inputs)  # Generate random data
    y = (np.sin(np.sum(X, axis=1)) + np.cos(np.prod(X, axis=1))).reshape(num_samples, 1)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return X, y

def train_neural_network_gradient(X, y, num_hidden_neurons, epochs=10000, learning_rate=0.5):
    input_layer_neurons = X.shape[1]  # Number of features
    output_neurons = 1  # Single output

    weights_input_hidden = np.random.uniform(size=(input_layer_neurons, num_hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(num_hidden_neurons, output_neurons))

    predicted_outputs = []

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        error = y - predicted_output
        d_predicted_output = error * predicted_output * (1 - predicted_output)

        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * hidden_layer_output * (1 - hidden_layer_output)

        # Update weights
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

        predicted_outputs.append(predicted_output.copy())

    return predicted_outputs

# Genetic Algorithm functions
def initialize_population(pop_size, weight_shape):
    return [np.random.uniform(-1, 1, weight_shape) for _ in range(pop_size)]

def fitness_function(y_true, y_pred):
    return -np.mean((y_true - y_pred) ** 2)

def crossover(parent1, parent2):
    child = parent1.copy()
    mask = np.random.rand(*parent1.shape) > 0.5
    child[mask] = parent2[mask]
    return child

def mutate(weights, mutation_rate):
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate
    weights[mutation_mask] += np.random.randn(np.sum(mutation_mask))
    return weights

def train_neural_network_ga(X, y, num_hidden_neurons, generations=1000, pop_size=50, mutation_rate=0.01):
    input_layer_neurons = X.shape[1]
    output_neurons = 1

    # Total weights = weights_input_hidden + weights_hidden_output
    weight_shape = (input_layer_neurons * num_hidden_neurons + num_hidden_neurons * output_neurons,)

    population = initialize_population(pop_size, weight_shape)

    best_outputs = []
    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            # Decode weights
            idx = 0
            w_ih = individual[idx:idx + input_layer_neurons * num_hidden_neurons].reshape((input_layer_neurons, num_hidden_neurons))
            idx += input_layer_neurons * num_hidden_neurons
            w_ho = individual[idx:].reshape((num_hidden_neurons, output_neurons))

            # Forward pass
            hidden_layer_output = sigmoid(np.dot(X, w_ih))
            predicted_output = sigmoid(np.dot(hidden_layer_output, w_ho))

            # Fitness calculation
            fitness = fitness_function(y, predicted_output)
            fitness_scores.append((fitness, individual, predicted_output))

        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        best_individual = fitness_scores[0][1]
        best_predicted_output = fitness_scores[0][2]
        best_outputs.append(best_predicted_output.copy())

        # Selection (top 50%)
        selected = [individual for _, individual, _ in fitness_scores[:pop_size // 2]]

        # Crossover & Mutation
        children = []
        while len(children) < pop_size:
            parents = random.sample(selected, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, mutation_rate)
            children.append(child)

        population = children

    return best_outputs

def unflatten_weights(w_flat, input_dim, hidden_dim, output_dim=1):
    start = 0
    end = input_dim * hidden_dim
    w_ih = w_flat[start:end].reshape((input_dim, hidden_dim))
    
    start = end
    end = end + hidden_dim * output_dim
    w_ho = w_flat[start:end].reshape((hidden_dim, output_dim))
    
    return w_ih, w_ho

def flatten_weights(w_ih, w_ho):
    return np.concatenate([w_ih.ravel(), w_ho.ravel()])

def forward_and_gradient(W_flat, input_dim, num_hidden_neurons, X, y):
    # Unpack weights
    w_ih, w_ho = unflatten_weights(W_flat, input_dim, num_hidden_neurons, output_dim=1)
    
    # Forward pass
    hidden_input = np.dot(X, w_ih)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w_ho)
    prediction = sigmoid(final_input)
    
    # Mean-squared error
    error = y - prediction
    mse = np.mean(error**2)
    
    # Backprop to get gradient
    # Output layer delta
    d_out = error * prediction * (1 - prediction)  # shape: (num_samples, 1)
    
    # Hidden layer delta
    error_hidden = d_out.dot(w_ho.T)  # shape: (num_samples, num_hidden_neurons)
    d_hidden = error_hidden * hidden_output * (1 - hidden_output)
    
    # Grad w.r.t. w_ho
    grad_w_ho = hidden_output.T.dot(d_out) / X.shape[0]  # average
    # Grad w.r.t. w_ih
    grad_w_ih = X.T.dot(d_hidden) / X.shape[0]  # average
    
    # Flatten the gradient
    grad_flat = flatten_weights(grad_w_ih, grad_w_ho)
    
    return mse, grad_flat

def golden_ratio_search(
    W, 
    p, 
    input_dim, 
    num_hidden_neurons, 
    X, 
    y, 
    a=1e-8, 
    b=1.0, 
    tol=1e-6, 
    max_iters=50
):
    """
    Perform a golden ratio (golden section) search to find the alpha that minimizes MSE(W + alpha * p).
    :param W: Current flattened weight vector
    :param p: Conjugate direction (flattened)
    :param input_dim: Number of input neurons
    :param num_hidden_neurons: Number of hidden neurons
    :param X: Training data
    :param y: Target data
    :param a: Left boundary of alpha
    :param b: Right boundary of alpha
    :param tol: Tolerance for stopping
    :param max_iters: Maximum search iterations
    :return: alpha_star that (approximately) minimizes the MSE
    """
    # Golden ratio constant
    phi = 0.6180339887498949  # ~ (sqrt(5)-1)/2
    c = 1.0 - phi  # This is the '1 - phi' portion

    # Helper to compute MSE for a given alpha
    def mse_at_alpha(alpha):
        W_test = W + alpha * p
        mse_val, _ = forward_and_gradient(W_test, input_dim, num_hidden_neurons, X, y)
        return mse_val

    # Initialize the two internal test points
    alpha1 = a + c * (b - a)
    alpha2 = b - c * (b - a)

    f1 = mse_at_alpha(alpha1)
    f2 = mse_at_alpha(alpha2)

    for _ in range(max_iters):
        # Narrow down [a, b] depending on which point is better
        if f1 > f2:
            # Minimum in (alpha1, b)
            a = alpha1
            alpha1 = alpha2
            f1 = f2
            alpha2 = b - c * (b - a)
            f2 = mse_at_alpha(alpha2)
        else:
            # Minimum in (a, alpha2)
            b = alpha2
            alpha2 = alpha1
            f2 = f1
            alpha1 = a + c * (b - a)
            f1 = mse_at_alpha(alpha1)

        # Stopping criterion if interval is sufficiently small
        if abs(b - a) < tol:
            break

    # Return the midpoint of our final bracket
    return 0.5 * (a + b)

def train_neural_network_conjugate_gradient(X, y, num_hidden_neurons, epochs=100, tol=1e-6):
    input_dim = X.shape[1]
    output_dim = 1
    
    w_ih_init = np.random.uniform(size=(input_dim, num_hidden_neurons))
    w_ho_init = np.random.uniform(size=(num_hidden_neurons, output_dim))
    W = flatten_weights(w_ih_init, w_ho_init)  # Flattened weight vector

    _, G = forward_and_gradient(W, input_dim, num_hidden_neurons, X, y)
    p = -G  # initial direction = -grad
    G_norm_sq = np.dot(G, G)
    
    predicted_outputs = []
    
    for epoch in range(epochs):
        best_alpha = golden_ratio_search(
            W=W,
            p=p,
            input_dim=input_dim,
            num_hidden_neurons=num_hidden_neurons,
            X=X,
            y=y,
            a=1e-8,      # can be smaller or bigger as needed
            b=1.0,       # upper bound for alpha
            tol=1e-7,    # tolerance for golden search
            max_iters=500 # number of golden search steps
        )

        # Update weights
        W_new = W + best_alpha * p
        
        # Compute new gradient
        mse_new, G_new = forward_and_gradient(W_new, input_dim, num_hidden_neurons, X, y)
        G_new_norm_sq = np.dot(G_new, G_new)
        
        # If gradient is small enough -> stop
        if np.sqrt(G_new_norm_sq) < tol:
            W = W_new
            break
        
        # Compute beta (Fletcher–Reeves)
        beta = G_new_norm_sq / G_norm_sq
        
        # Update direction
        p = -G_new + beta * p
        
        # Prepare next iteration
        W = W_new
        G = G_new
        G_norm_sq = G_new_norm_sq
        
        # Store the network's predictions for plotting/comparison
        # (Forward pass using the updated weights)
        w_ih_final, w_ho_final = unflatten_weights(W, input_dim, num_hidden_neurons, output_dim)
        hidden_output = sigmoid(X @ w_ih_final)
        final_output = sigmoid(hidden_output @ w_ho_final)
        predicted_outputs.append(final_output.copy())
    
    return predicted_outputs

def plot_results(y_true, y_pred_gradient, y_pred_ga, y_pred_cg, epoch):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true.flatten(), mode='lines+markers', name='True'))
    fig.add_trace(go.Scatter(y=y_pred_gradient.flatten(), mode='lines+markers', name='Gradient Descent'))
    fig.add_trace(go.Scatter(y=y_pred_ga.flatten(), mode='lines+markers', name='Genetic Algorithm'))
    fig.add_trace(go.Scatter(y=y_pred_cg.flatten(), mode='lines+markers', name='Conjugate Gradient'))
    fig.update_layout(title=f'Comparison at epoch {epoch}',
                      xaxis_title='Sample Index',
                      yaxis_title='Value',
                      legend_title='Legend')

    mae_gradient = np.mean(np.abs(y_true - y_pred_gradient))
    mae_ga = np.mean(np.abs(y_true - y_pred_ga))
    mae_cg = np.mean(np.abs(y_true - y_pred_cg))

    mse_gradient = np.mean((y_true - y_pred_gradient) ** 2)
    mse_ga = np.mean((y_true - y_pred_ga) ** 2)
    mse_cg = np.mean((y_true - y_pred_cg) ** 2)


    st.write(f'Epoch {epoch}')
    st.write(f'Gradient Descent Error: MAE = {mae_gradient}, MSE = {mse_gradient}')
    st.write(f'Genetic Algorithm Error: MAE = {mae_ga}, MSE = {mse_ga}')
    st.plotly_chart(fig)

    if len(y_true) <= 100:
        results_df = pd.DataFrame({
            'True Value': y_true.flatten(),
            'Gradient Descent': y_pred_gradient.flatten(),
            'Genetic Algorithm': y_pred_ga.flatten(),
            'Conjugate Gradient': y_pred_cg.flatten()
        })

        errors_df = pd.DataFrame({
            'Gradient MSE': mse_gradient.flatten(),
            'Genetic MSE': mse_ga.flatten(),
            'Conjugate MSE': mse_cg.flatten()
        })

        st.write("Detailed result:")
        st.dataframe(results_df)

        st.write("Errors:")
        st.dataframe(errors_df)


def main():
    random.seed(g_seed)
    np.random.seed(g_seed)
    
    st.title('Neural Network Training Comparison')

    num_samples = st.sidebar.number_input('Samples:', min_value=1, value=100, step=1)
    num_inputs = st.sidebar.number_input('Inputs (count):', min_value=1, value=3, step=1)
    num_hidden_neurons = st.sidebar.number_input('Hidden neurons:', min_value=1, value=4, step=1)
    learning_rate = st.sidebar.number_input('Learning rate:', min_value=0.0001, value=0.5, step=0.0001, format="%.4f")

    st.sidebar.write('### Gradient Descent Parameters')
    epochs = st.sidebar.number_input('Epochs (Gradient Descent):', min_value=1, value=1000, step=1)

    st.sidebar.write('### Genetic Algorithm Parameters')
    generations = st.sidebar.number_input('Generations (GA):', min_value=1, value=100, step=1)
    population_size = st.sidebar.number_input('Population Size:', min_value=2, value=50, step=1)
    mutation_rate = st.sidebar.number_input('Mutation Rate:', min_value=0.0001, value=0.01, step=0.0001, format="%.4f")

    if 'trained' not in st.session_state:
        st.session_state.trained = False

    if st.sidebar.button('Start learning'):
        X, y = generate_data(int(num_samples), int(num_inputs))

        # Gradient Descent Training
        predicted_outputs_gradient = train_neural_network_gradient(X, y, int(num_hidden_neurons), int(epochs), learning_rate)

        # Genetic Algorithm Training
        predicted_outputs_ga = train_neural_network_ga(X, y, int(num_hidden_neurons), int(generations), int(population_size), mutation_rate)

        predicted_outputs_cg = train_neural_network_conjugate_gradient(X, y, int(num_hidden_neurons), epochs=int(epochs), tol=1e-6)

        # Store in session state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.predicted_outputs_gradient = predicted_outputs_gradient
        st.session_state.predicted_outputs_ga = predicted_outputs_ga
        st.session_state.predicted_outputs_cg = predicted_outputs_cg
        st.session_state.epochs = int(epochs)
        st.session_state.generations = int(generations)
        st.session_state.trained = True

    if st.session_state.trained:
        epoch = st.slider('Epoch / Generation:', min_value=0, max_value=max(st.session_state.epochs, st.session_state.generations)-1, value=0, step=1)
        # Handle out-of-range indices
        epoch_gradient = min(epoch, len(st.session_state.predicted_outputs_gradient)-1)
        epoch_ga = min(epoch, len(st.session_state.predicted_outputs_ga)-1)

        y_pred_gradient = st.session_state.predicted_outputs_gradient[epoch_gradient]
        y_pred_cg = st.session_state.predicted_outputs_cg[epoch_gradient]
        y_pred_ga = st.session_state.predicted_outputs_ga[epoch_ga]

        plot_results(st.session_state.y, y_pred_gradient, y_pred_ga, y_pred_cg, epoch)
    else:
        st.write('Press "Start learning" to run the simulation')

if __name__ == "__main__":
    main()
