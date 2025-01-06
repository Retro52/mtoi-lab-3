import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

g_seed: int = 52

def generate_data(num_samples, num_inputs):
    X = np.random.rand(num_samples, num_inputs)  # Generate random data
    y = (np.sin(np.sum(X, axis=1))).reshape(num_samples, 1)
    # y = (np.sin(np.sum(X, axis=1)) + np.cos(np.prod(X, axis=1))).reshape(num_samples, 1)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    w_ih, w_ho = unflatten_weights(W_flat, input_dim, num_hidden_neurons, output_dim=1)
    
    # Forward pass
    hidden_input = np.dot(X, w_ih)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w_ho)
    prediction = sigmoid(final_input)
    
    error = y - prediction
    mse = np.mean(error**2)
    
    d_out = -2 * error * prediction * (1 - prediction)
    
    # Hidden layer delta
    error_hidden = d_out.dot(w_ho.T)
    d_hidden = error_hidden * hidden_output * (1 - hidden_output)
    
    # Grad w.r.t. w_ho
    grad_w_ho = hidden_output.T.dot(d_out) / X.shape[0]
    grad_w_ih = X.T.dot(d_hidden) / X.shape[0]
    
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
    phi = 0.6180339887498949
    c = 1.0 - phi

    # Helper to compute MSE for a given alpha
    def mse_at_alpha(alpha):
        W_test = W + alpha * p
        mse_val, _ = forward_and_gradient(W_test, input_dim, num_hidden_neurons, X, y)
        return mse_val

    alpha1 = a + c * (b - a)
    alpha2 = b - c * (b - a)

    f1 = mse_at_alpha(alpha1)
    f2 = mse_at_alpha(alpha2)

    for _ in range(max_iters):
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

        if abs(b - a) < tol:
            break

    return 0.5 * (a + b)

def train_neural_network_conjugate_gradient(X, y, num_hidden_neurons, epochs=100, tol=1e-6, iter_change_dir=5, min_alpha=1e-3):
    input_dim = X.shape[1]
    output_dim = 1
    
    w_ih_init = np.random.uniform(size=(input_dim, num_hidden_neurons))
    w_ho_init = np.random.uniform(size=(num_hidden_neurons, output_dim))
    W = flatten_weights(w_ih_init, w_ho_init)

    _, G = forward_and_gradient(W, input_dim, num_hidden_neurons, X, y)
    p = -G
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
            a=min_alpha,
            b=1.0,
            tol=1e-7,
            max_iters=500
        )

        W_new = W + best_alpha * p
        
        _, G_new = forward_and_gradient(W_new, input_dim, num_hidden_neurons, X, y)
        G_new_norm_sq = np.dot(G_new, G_new)
        
        # Check convergence
        if np.sqrt(G_new_norm_sq) < tol:
            W = W_new
            break
        
        # Shuffle the direction each 'iter_change_dir' iterations to avoid being stuck at locals
        if iter_change_dir > 0 and epoch % iter_change_dir == 0:
            p = -G_new
        else:
            beta = G_new_norm_sq / G_norm_sq
            p = -G_new + beta * p
        
        W = W_new
        G = G_new
        G_norm_sq = G_new_norm_sq
        
        # (Forward pass using the updated weights)
        w_ih_final, w_ho_final = unflatten_weights(W, input_dim, num_hidden_neurons, output_dim)
        hidden_output = sigmoid(X @ w_ih_final)
        final_output = sigmoid(hidden_output @ w_ho_final)
        predicted_outputs.append(final_output.copy())
    
    return predicted_outputs

def train_anfis(X, y, alpha=0.05, beta=0.1, gamma=0.05, epochs=1000):
    cA1, cA2 = 0.25, 0.75
    sA1, sA2 = 0.1, 0.1

    cB1, cB2 = 0.25, 0.75
    sB1, sB2 = 0.1, 0.1

    a1, b1, r1 = 0.1, 0.1, 0.0
    a2, b2, r2 = 0.1, 0.1, 0.0

    y = y.reshape(-1, 1)  # ensure column vector

    # helper - gaussian membership function
    def gauss_mf(x, c, s):
        # +1e-12 in the denominator to avoid division by 0
        return np.exp(-((x - c) ** 2) / (s**2 + 1e-12))

    predicted_outputs = []

    for epoch in range(epochs):
        # Forward pass
        muA1 = gauss_mf(X[:, 0], cA1, sA1)
        muA2 = gauss_mf(X[:, 0], cA2, sA2)
        muB1 = gauss_mf(X[:, 1], cB1, sB1)
        muB2 = gauss_mf(X[:, 1], cB2, sB2)

        # Firing strengths
        w1 = muA1 * muB1
        w2 = muA2 * muB2

        # Normalized firing strengths
        w_sum = w1 + w2 + 1e-12
        w1n = w1 / w_sum
        w2n = w2 / w_sum

        # Sugeno outputs f1, f2
        # f1 = a1*x + b1*y + r1
        # f2 = a2*x + b2*y + r2
        f1 = a1 * X[:, 0] + b1 * X[:, 1] + r1
        f2 = a2 * X[:, 0] + b2 * X[:, 1] + r2

        # Overall output
        out = w1n * f1 + w2n * f2
        out = out.reshape(-1, 1)

        # Error and MSE
        error = (y - out)
        mse = np.mean(error**2)

        # Backward pass (Gradient Computation)
        d_a1 = -np.mean(error * w1n.reshape(-1, 1) * X[:, 0].reshape(-1, 1))
        d_b1 = -np.mean(error * w1n.reshape(-1, 1) * X[:, 1].reshape(-1, 1))
        d_r1 = -np.mean(error * w1n.reshape(-1, 1))

        d_a2 = -np.mean(error * w2n.reshape(-1, 1) * X[:, 0].reshape(-1, 1))
        d_b2 = -np.mean(error * w2n.reshape(-1, 1) * X[:, 1].reshape(-1, 1))
        d_r2 = -np.mean(error * w2n.reshape(-1, 1))

        # Helper for membership parameter partials
        def firing_norm_derivs(dw1, dw2, w1, w2, w_sum):
            # shape-wise: dw1, dw2, w1, w2, w_sum are all (N,)
            dwsum = dw1 + dw2
            dw1n = (dw1 * w_sum - w1 * dwsum) / (w_sum**2)
            dw2n = (dw2 * w_sum - w2 * dwsum) / (w_sum**2)
            return dw1n, dw2n


        # --- partials wrt cA1, sA1 (affect w1) ---
        x0 = X[:, 0]
        x1 = X[:, 1]

        dmuA1_dcA1 = muA1 * (2.0 * (x0 - cA1) / (sA1**2 + 1e-12))
        dmuA1_dsA1 = muA1 * (2.0 * (x0 - cA1)**2 / (sA1**3 + 1e-12))

        dw1_dcA1 = dmuA1_dcA1 * muB1
        dw1_dsA1 = dmuA1_dsA1 * muB1

        # w2 doesn't depend on A1 => dw2_dcA1 = 0, dw2_dsA1 = 0
        dw1n_dcA1, dw2n_dcA1 = firing_norm_derivs(dw1_dcA1, 0, w1, w2, w_sum)
        dw1n_dsA1, dw2n_dsA1 = firing_norm_derivs(dw1_dsA1, 0, w1, w2, w_sum)

        # cA1 => dOut/dcA1 = dw1n_dcA1*f1 + dw2n_dcA1*f2
        dOut_dcA1 = dw1n_dcA1 * f1 + dw2n_dcA1 * f2
        dOut_dsA1 = dw1n_dsA1 * f1 + dw2n_dsA1 * f2

        # dE/dcA1 = - sum( error * dOut_dcA1 ) / N
        d_cA1 = -np.mean(error.reshape(-1) * dOut_dcA1)
        d_sA1 = -np.mean(error.reshape(-1) * dOut_dsA1)

        # --- partials wrt cA2, sA2 (affect w2) ---
        dmuA2_dcA2 = muA2 * (2.0 * (x0 - cA2) / (sA2**2 + 1e-12))
        dmuA2_dsA2 = muA2 * (2.0 * (x0 - cA2)**2 / (sA2**3 + 1e-12))

        dw2_dcA2 = dmuA2_dcA2 * muB2
        dw2_dsA2 = dmuA2_dsA2 * muB2

        # w1 doesn't depend on A2 => dw1_dcA2=0, dw1_dsA2=0
        dw1n_dcA2, dw2n_dcA2 = firing_norm_derivs(0, dw2_dcA2, w1, w2, w_sum)
        dw1n_dsA2, dw2n_dsA2 = firing_norm_derivs(0, dw2_dsA2, w1, w2, w_sum)

        dOut_dcA2 = dw1n_dcA2 * f1 + dw2n_dcA2 * f2
        dOut_dsA2 = dw1n_dsA2 * f1 + dw2n_dsA2 * f2

        d_cA2 = -np.mean(error.reshape(-1) * dOut_dcA2)
        d_sA2 = -np.mean(error.reshape(-1) * dOut_dsA2)

        # --- partials wrt cB1, sB1 (affect w1) ---
        dmuB1_dcB1 = muB1 * (2.0 * (x1 - cB1) / (sB1**2 + 1e-12))
        dmuB1_dsB1 = muB1 * (2.0 * (x1 - cB1)**2 / (sB1**3 + 1e-12))

        dw1_dcB1 = muA1 * dmuB1_dcB1
        dw1_dsB1 = muA1 * dmuB1_dsB1

        # w2 doesn't depend on B1 => 0
        dw1n_dcB1, dw2n_dcB1 = firing_norm_derivs(dw1_dcB1, 0, w1, w2, w_sum)
        dw1n_dsB1, dw2n_dsB1 = firing_norm_derivs(dw1_dsB1, 0, w1, w2, w_sum)

        dOut_dcB1 = dw1n_dcB1 * f1 + dw2n_dcB1 * f2
        dOut_dsB1 = dw1n_dsB1 * f1 + dw2n_dsB1 * f2

        d_cB1 = -np.mean(error.reshape(-1) * dOut_dcB1)
        d_sB1 = -np.mean(error.reshape(-1) * dOut_dsB1)

        # --- partials wrt cB2, sB2 (affect w2) ---
        dmuB2_dcB2 = muB2 * (2.0 * (x1 - cB2) / (sB2**2 + 1e-12))
        dmuB2_dsB2 = muB2 * (2.0 * (x1 - cB2)**2 / (sB2**3 + 1e-12))

        dw2_dcB2 = muA2 * dmuB2_dcB2
        dw2_dsB2 = muA2 * dmuB2_dsB2

        dw1n_dcB2, dw2n_dcB2 = firing_norm_derivs(0, dw2_dcB2, w1, w2, w_sum)
        dw1n_dsB2, dw2n_dsB2 = firing_norm_derivs(0, dw2_dsB2, w1, w2, w_sum)

        dOut_dcB2 = dw1n_dcB2 * f1 + dw2n_dcB2 * f2
        dOut_dsB2 = dw1n_dsB2 * f1 + dw2n_dsB2 * f2

        d_cB2 = -np.mean(error.reshape(-1) * dOut_dcB2)
        d_sB2 = -np.mean(error.reshape(-1) * dOut_dsB2)

        # membership centers/widths
        cA1 -= alpha * d_cA1
        sA1 -= beta  * d_sA1
        cA2 -= alpha * d_cA2
        sA2 -= beta  * d_sA2
        cB1 -= alpha * d_cB1
        sB1 -= beta  * d_sB1
        cB2 -= alpha * d_cB2
        sB2 -= beta  * d_sB2

        # linear parameters
        a1  -= gamma * d_a1
        b1  -= gamma * d_b1
        r1  -= gamma * d_r1
        a2  -= gamma * d_a2
        b2  -= gamma * d_b2
        r2  -= gamma * d_r2

        predicted_outputs.append(out.copy())

    return predicted_outputs

def plot_results(y_true, y_pred_gradient, y_pred_ga, y_pred_cg, y_pred_anfis, epoch):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true.flatten(),          mode='lines+markers', name='True'))
    fig.add_trace(go.Scatter(y=y_pred_gradient.flatten(), mode='lines+markers', name='Gradient Descent'))
    fig.add_trace(go.Scatter(y=y_pred_ga.flatten(),       mode='lines+markers', name='Genetic Algorithm'))
    fig.add_trace(go.Scatter(y=y_pred_cg.flatten(),       mode='lines+markers', name='Conjugate Gradient'))
    fig.add_trace(go.Scatter(y=y_pred_anfis.flatten(),    mode='lines+markers', name='ANFIS'))
    
    fig.update_layout(title=f'Comparison at epoch {epoch}',
                      xaxis_title='Sample Index',
                      yaxis_title='Value',
                      legend_title='Legend')

    mae_gradient = np.mean(np.abs(y_true - y_pred_gradient))
    mae_ga = np.mean(np.abs(y_true - y_pred_ga))
    mae_cg = np.mean(np.abs(y_true - y_pred_cg))
    mae_anfis = np.mean(np.abs(y_true - y_pred_anfis))

    mse_gradient = np.mean((y_true - y_pred_gradient) ** 2)
    mse_ga = np.mean((y_true - y_pred_ga) ** 2)
    mse_cg = np.mean((y_true - y_pred_cg) ** 2)
    mse_anfis = np.mean((y_true - y_pred_anfis) ** 2)


    st.write(f'Epoch {epoch}')
    st.write(f'Gradient Descent Error: MAE = {mae_gradient}, MSE = {mse_gradient}')
    st.write(f'Genetic Algorithm Error: MAE = {mae_ga}, MSE = {mse_ga}')
    st.write(f'Conjugate Gradient Error: MAE = {mae_cg}, MSE = {mse_cg}')
    st.write(f'ANFIS Error: MAE = {mae_anfis}, MSE = {mse_anfis}')
    st.plotly_chart(fig)

    if len(y_true) <= 100:
        results_df = pd.DataFrame({
            'True Value': y_true.flatten(),
            'Gradient Descent': y_pred_gradient.flatten(),
            'Genetic Algorithm': y_pred_ga.flatten(),
            'Conjugate Gradient': y_pred_cg.flatten(),
            'ANFIS': y_pred_anfis.flatten()
        })

        errors_df = pd.DataFrame({
            'Gradient MSE': mse_gradient.flatten(),
            'Genetic MSE': mse_ga.flatten(),
            'Conjugate MSE': mse_cg.flatten(),
            'ANFIS': mse_anfis.flatten()
        })

        st.write("Detailed result:")
        st.dataframe(results_df)

        st.write("Errors:")
        st.dataframe(errors_df)


def main():
    random.seed(g_seed)
    np.random.seed(g_seed)
    
    st.title('Neural Network Training Comparison')
    num_samples        = st.sidebar.number_input('Samples:',        min_value=1,      value=100,  step=1)
    num_inputs         = st.sidebar.number_input('Inputs (count):', min_value=1,      value=3,    step=1)
    num_hidden_neurons = st.sidebar.number_input('Hidden neurons:', min_value=1,      value=4,    step=1)
    learning_rate      = st.sidebar.number_input('Learning rate:',  min_value=0.0001, value=0.5,  step=0.0001, format="%.4f")

    st.sidebar.write('### Genetic Algorithm Parameters')
    generations     = st.sidebar.number_input('Generations (GA):', min_value=1,      value=1000, step=1)
    population_size = st.sidebar.number_input('Population Size:',  min_value=2,      value=50,   step=1)
    mutation_rate   = st.sidebar.number_input('Mutation Rate:',    min_value=0.0001, value=0.01, step=0.0001, format="%.4f")

    st.sidebar.write('### Gradient Descent Parameters')
    epochs = st.sidebar.number_input('Epochs (Gradient Descent):', min_value=1, value=1000, step=1)

    st.sidebar.write('### Conjugate Algorithm Parameters')
    cg_dir_change = st.sidebar.number_input('Epochs before dir change:', min_value=1,        value=20,     step=1)
    cg_min_alpha  = st.sidebar.number_input('Min alpha value:',          min_value=0.0,      value=1e-4,   step=1e-5, format="%.6f")

    st.sidebar.write('### ANFIS Parameters')
    alpha_val    = st.sidebar.number_input('Alpha (membership centers)', min_value=0.0001, value=0.05, step=0.0001, format="%.4f")
    beta_val     = st.sidebar.number_input('Beta (membership widths)',   min_value=0.0001, value=0.10, step=0.0001, format="%.4f")
    gamma_val    = st.sidebar.number_input('Gamma (linear coeffs)',      min_value=0.0001, value=0.05, step=0.0001, format="%.4f")
    anfis_epochs = st.sidebar.number_input('Epochs (ANFIS):',            min_value=1, value=1000, step=1)

    if 'trained' not in st.session_state:
        st.session_state.trained = False

    if st.sidebar.button('Start learning'):
        X, y = generate_data(int(num_samples), int(num_inputs))

        predicted_outputs_gradient = train_neural_network_gradient(X, y, int(num_hidden_neurons), int(epochs), learning_rate)
        predicted_outputs_ga       = train_neural_network_ga(X, y, int(num_hidden_neurons), int(generations), int(population_size), mutation_rate)
        predicted_outputs_cg       = train_neural_network_conjugate_gradient(X, y, int(num_hidden_neurons), epochs=int(epochs), tol=1e-6, iter_change_dir=cg_dir_change, min_alpha=cg_min_alpha)
        predicted_outputs_anfis    = train_anfis(X, y, alpha=alpha_val, beta=beta_val, gamma=gamma_val, epochs=int(anfis_epochs))

        st.session_state.X = X
        st.session_state.y = y
        st.session_state.predicted_outputs_anfis = predicted_outputs_anfis
        st.session_state.predicted_outputs_gradient = predicted_outputs_gradient
        st.session_state.predicted_outputs_ga = predicted_outputs_ga
        st.session_state.predicted_outputs_cg = predicted_outputs_cg
        st.session_state.epochs = int(epochs)
        st.session_state.epochs_anfis = int(anfis_epochs)
        st.session_state.generations = int(generations)
        st.session_state.trained = True

    if st.session_state.trained:
        epoch = st.slider('Epoch / Generation:', min_value=0, max_value=max(st.session_state.epochs, st.session_state.generations)-1, value=0, step=1)
        # Handle out-of-range indices
        epoch_gradient = min(epoch, len(st.session_state.predicted_outputs_gradient)-1)
        epoch_ga = min(epoch, len(st.session_state.predicted_outputs_ga)-1)
        epoch_anfis = min(epoch, len(st.session_state.predicted_outputs_anfis)-1)

        y_pred_gradient = st.session_state.predicted_outputs_gradient[epoch_gradient]
        y_pred_cg = st.session_state.predicted_outputs_cg[epoch_gradient]
        y_pred_ga = st.session_state.predicted_outputs_ga[epoch_ga]
        y_pred_anfis = st.session_state.predicted_outputs_anfis[epoch_anfis]

        plot_results(st.session_state.y, y_pred_gradient, y_pred_ga, y_pred_cg, y_pred_anfis, epoch)
    else:
        st.write('Press "Start learning" to run the simulation')

if __name__ == "__main__":
    main()
