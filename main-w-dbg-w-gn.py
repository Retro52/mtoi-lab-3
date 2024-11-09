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

def plot_results(y_true, y_pred_gradient, y_pred_ga, epoch):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true.flatten(), mode='lines+markers', name='True'))
    fig.add_trace(go.Scatter(y=y_pred_gradient.flatten(), mode='lines+markers', name='Gradient Descent'))
    fig.add_trace(go.Scatter(y=y_pred_ga.flatten(), mode='lines+markers', name='Genetic Algorithm'))
    fig.update_layout(title=f'Comparison at epoch {epoch}',
                      xaxis_title='Sample Index',
                      yaxis_title='Value',
                      legend_title='Legend')

    mae_gradient = np.mean(np.abs(y_true - y_pred_gradient))
    mae_ga = np.mean(np.abs(y_true - y_pred_ga))

    mse_gradient = np.mean((y_true - y_pred_gradient) ** 2)
    mse_ga = np.mean((y_true - y_pred_ga) ** 2)


    st.write(f'Epoch {epoch}')
    st.write(f'Gradient Descent Error: MAE = {mae_gradient}, MSE = {mse_gradient}')
    st.write(f'Genetic Algorithm Error: MAE = {mae_ga}, MSE = {mse_ga}')
    st.plotly_chart(fig)

    if len(y_true) <= 100:
        results_df = pd.DataFrame({
            'True Value': y_true.flatten(),
            'Gradient Descent': y_pred_gradient.flatten(),
            'Genetic Algorithm': y_pred_ga.flatten()
        })

        errors_df = pd.DataFrame({
            'Gradient MSE': mse_gradient.flatten(),
            'Genetic MSE': mse_ga.flatten()
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

        # Store in session state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.predicted_outputs_gradient = predicted_outputs_gradient
        st.session_state.predicted_outputs_ga = predicted_outputs_ga
        st.session_state.epochs = int(epochs)
        st.session_state.generations = int(generations)
        st.session_state.trained = True

    if st.session_state.trained:
        epoch = st.slider('Epoch / Generation:', min_value=0, max_value=max(st.session_state.epochs, st.session_state.generations)-1, value=0, step=1)
        # Handle out-of-range indices
        epoch_gradient = min(epoch, len(st.session_state.predicted_outputs_gradient)-1)
        epoch_ga = min(epoch, len(st.session_state.predicted_outputs_ga)-1)

        y_pred_gradient = st.session_state.predicted_outputs_gradient[epoch_gradient]
        y_pred_ga = st.session_state.predicted_outputs_ga[epoch_ga]

        plot_results(st.session_state.y, y_pred_gradient, y_pred_ga, epoch)
    else:
        st.write('Press "Start learning" to run the simulation')

if __name__ == "__main__":
    main()
