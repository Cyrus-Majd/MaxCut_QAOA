import itertools
import time

import numpy as np
import networkx
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import cirq

def main(repetitions=10, maxiter=10):
    # for measuring time of program.
    startTime = time.time()

    # Set problem parameters
    n=3     # number of nodes
    p=5     # number of layers

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # # Recreate the minimal 3-node example from class
    # graph = networkx.Graph()
    # graph.add_nodes_from([0,1,2,3,4,5,6,7,8,9])
    # graph.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])
    # networkx.draw(graph)
    # # plt.show()    # GRAPHS THE NODES
    # # quit()

    # Each node in n nodes of the MAX-CUT graph corresponds to one of n qubits in the quantum circuit.
    # The state vector across the qubits encodes a node partitioning
    qubits = cirq.LineQubit.range(n)

    hadamard_circuit = cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits)
    )
    simulator = cirq.Simulator()
    print ("Put the initial state vector in a superposition of all possible node partitionings:")
    print (simulator.simulate(hadamard_circuit, initial_state=0))
    # quit()

    # Print an example circuit
    # Provide classical parameters such that the classical computer can control quantum partitioning
    betas = np.random.uniform(0, np.pi, size=p)
    gammas = np.random.uniform(0, 2*np.pi, size=p)
    circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)
    print('Example QAOA circuit:')
    print(circuit.to_text_diagram(transpose=True))
    # quit()

    # Create variables to store the largest cut and cut value found
    largest_cut_found = None
    largest_cut_value_found = 0

    # Initialize simulator
    simulator = cirq.Simulator()

    # For visualizing the optimization iterations
    betas_sched = []
    gammas_sched = []
    measurement_probability_distribution_sched = []
    cut_mean_sched = []

    # Define objective function (we'll use the negative expected cut value)
    def f(x):
        # Create circuit
        betas = x[:p]
        gammas = 2*x[p:]
        betas_sched.append(betas)
        gammas_sched.append(gammas)

        # Perform a series of operations parameterized by classical parameters
        # such that the final state vector is a superposition of good partitionings
        circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)

        # we also want a simulation where the final quantum state vector is not collapsed via measurement
        circuit_no_measurement = qaoa_max_cut_circuit_no_measurement(qubits, betas, gammas, graph)
        final_state_vector = simulator.simulate(circuit_no_measurement).final_state_vector
        print("The final quantum state vector without measurement collapse:")
        print(final_state_vector)
        # quit()

        # from amplitudes to measurement probabilities
        measurement_probability_distribution_sched.append(np.square(np.absolute(final_state_vector)))

        # Sample bitstrings from circuit
        result = simulator.run(circuit, repetitions=repetitions)
        bitstrings = result.measurements['m']
        # Process bitstrings
        nonlocal largest_cut_found
        nonlocal largest_cut_value_found
        values = cut_values(bitstrings, graph)
        max_value_index = np.argmax(values)
        max_value = values[max_value_index]
        if max_value > largest_cut_value_found:
            largest_cut_value_found = max_value
            largest_cut_found = bitstrings[max_value_index]
        mean = np.mean(values)
        cut_mean_sched.append(mean)

        return -mean

    # Provide classical parameters such that the classical computer can control quantum partitioning
    # Pick an initial guess
    x0 = np.random.uniform(0, np.pi, size=2 * p)

    # Optimize for a good set of gammas and betas
    # Optimize f
    print('Optimizing objective function ...')
    scipy.optimize.minimize(f,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': maxiter})

    # Compute best possible cut value via brute force search
    all_bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    all_values = cut_values(all_bitstrings, graph)
    max_cut_value = np.max(all_values)

    # Print the results
    print(f'The largest cut value found was {largest_cut_value_found}.')
    print(f'The largest possible cut has size {max_cut_value}.')
    print(f'The approximation ratio achieved is {largest_cut_value_found / max_cut_value}.')

    # Visualize the data
    fig, ((bax,gax),(dax,max)) = plt.subplots(2,2)

    # Create a color coding and legend
    max_opt_iter = len(measurement_probability_distribution_sched)
    colors = plt.cm.Wistia(np.linspace(0,1,max_opt_iter))
    custom_legend = [Line2D([0], [0], color=colors[0]),
                     Line2D([0], [0], color=colors[-1])]

    # Plot gammas and betas as function of optimization iteration
    qaoa_layers = [i for i in range(p)]

    bax.set_title('Beta optimization')
    bax.set_ylabel('Radians')
    bax.xaxis.set_major_locator(MaxNLocator(integer=True))
    bax.set_xlabel('QAOA layer')

    gax.set_title('Gamma optimization')
    gax.set_ylabel('Radians')
    gax.xaxis.set_major_locator(MaxNLocator(integer=True))
    gax.set_xlabel('QAOA layer')

    for opt_iter, (layer_betas,layer_gammas) in enumerate(zip(betas_sched,gammas_sched)):
        bax.plot(qaoa_layers, layer_betas,  color=colors[opt_iter])
        gax.plot(qaoa_layers, layer_gammas, color=colors[opt_iter])
    bax.legend(custom_legend, ['Begin', 'End'], loc='lower right')
    gax.legend(custom_legend, ['Begin', 'End'], loc='lower right')

    # Plot measurement_probability_distribution_sched as function of optimization iteration
    basis_states = [i for i in range(1<<n)]
    dax.set_title('Measurement distribution')
    dax.set_ylabel('Measurement probability')
    dax.set_xlabel('Basis state')
    dax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:03b}"))
    dax.xaxis.set_ticks(np.arange(0, 111, 1))
    for opt_iter, histogram in enumerate(measurement_probability_distribution_sched):
        dax.plot(basis_states, histogram, color=colors[opt_iter])
    dax.legend(custom_legend, ['Begin', 'End'], loc='lower right')

    # Plot mean cut value vs. optimization iteration
    max.set_title('Cut value optimization')
    max.set_ylabel('Cut value')
    max.set_xlabel('Optimization iteration')
    max.plot(cut_mean_sched,color=colors[max_opt_iter//2],label='QAOA mean cut value')
    max.plot([max_cut_value for i in range(max_opt_iter)],color=colors[-1],label='True max-cut value')
    max.legend(loc='lower right')

    endTime = time.time()

    print("ELAPSED TIME:", endTime - startTime)

    plt.tight_layout()
    plt.show()


def rzz(rads):
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    return cirq.ZZPowGate(exponent=rads / np.pi, global_shift=-1)


def qaoa_max_cut_unitary(qubits, betas, gammas, graph):  # Nodes should be integers
    for beta, gamma in zip(betas, gammas):
        # Need an operator (quantum gate) that encodes an edge
        # https://quantumai.google/reference/python/cirq/ops/ZZPowGate
        yield (rzz(gamma).on(qubits[i], qubits[j]) for i, j in graph.edges)
        # https://quantumai.google/reference/python/cirq/ops/rx
        yield cirq.rx(2 * beta).on_each(*qubits)


def qaoa_max_cut_circuit(qubits, betas, gammas, graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph),
        # Measure
        cirq.measure(*qubits, key='m'))


def qaoa_max_cut_circuit_no_measurement(qubits, betas, gammas, graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph))


def cut_values(bitstrings, graph):
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1)**bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return vals


if __name__ == '__main__':
    main()