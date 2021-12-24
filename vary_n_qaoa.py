
## EDITS TO ORIGINAL CODE DONE BY CYRUS MAJD

"""Runs the Quantum Approximate Optimization Algorithm on Max-Cut.
    THIS VERSION CHANGES THE VALUE OF N AS THE SIMULATION GOES ON

=== EXAMPLE OUTPUT ===

Example QAOA circuit:
0         1         2
│         │         │
H         H         H
│         │         │
ZZ────────ZZ^0.974  │
│         │         │
Rx(0.51π) ZZ────────ZZ^0.974
│         │         │
│         Rx(0.51π) Rx(0.51π)
│         │         │
M('m')────M─────────M
│         │         │
Optimizing objective function ...
The largest cut value found was 2.0.
The largest possible cut has size 2.0.
The approximation ratio achieved is 1.0.
"""

import itertools

import numpy as np
import networkx
import scipy.optimize
import matplotlib.pyplot as plt
import statistics

import cirq

def main(repetitions=500, maxiter=128):

    # set array for y values for the plot.
    yValues = []

    # set trials range
    numOfTrials = 16

    # Set problem parameters
    n=3
    max_p=10

    # For plotting approximation ratios as a function of varying p
    approx_ratios_dict = {}
    for p in range(max_p,max_p+1):
        # For each key p we will store a set of values from trials
        approx_ratios_dict[p]=set()

    # Each trial uses a new randomly generated problem instance
    for trial in range(numOfTrials):

        print("TRIAL IS FINISHED!!!! TRIAL NUMBER ", trial)
        n = n + 1

        # Generate a random 3-regular graph on n nodes
        graph = networkx.random_regular_graph(2, n)

        for p in range(max_p,max_p+1):

            # Each node in n nodes of the MAX-CUT graph corresponds to one of n qubits in the quantum circuit.
            # The state vector across the qubits encodes a node partitioning
            qubits = cirq.LineQubit.range(n)

            # Create variables to store the largest cut and cut value found
            largest_cut_found = None
            largest_cut_value_found = 0
            final_mean = None

            # Initialize simulator
            simulator = cirq.Simulator()

            # Define objective function (we'll use the negative expected cut value)
            def f(x):
                # Create circuit
                betas = x[:p]
                gammas = 2*x[p:]

                # Perform a series of operations parameterized by classical parameters
                # such that the final state vector is a superposition of good partitionings
                circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)

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
                nonlocal final_mean
                final_mean = mean

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

            approx_ratios_dict[p].add( final_mean / max_cut_value )

            # Print the results
            print('trial={},p={}'.format(trial,p))
            print('The largest cut value found was {}.'.format(largest_cut_value_found))
            print('The largest possible cut has size {}.'.format(max_cut_value))
            print("APPROXIMATION RATIO FOR N=", n, " AND P=", p, ":", (final_mean/max_cut_value))
            yValues.append((final_mean/max_cut_value))

    # fig, ax = plt.subplots()
    # ax.set_title('Approximation ratio vs. p')
    # ax.set_ylabel('Approximation ratio')
    # ax.set_xlabel('p')
    # ps, approx_ratios = zip(*sorted(approx_ratios_dict.items())) # unpack a list of pairs into two tuples
    # ax.errorbar(ps, [statistics.mean(set) for set in approx_ratios],
    #             yerr=[statistics.stdev(set) for set in approx_ratios])
    # plt.tight_layout()
    # plt.show()

    tmp = []
    for i in range(n-3):
        tmp.append(i)

    print(tmp)
    print(yValues)
    ax = plt.gca()
    ax.set_ylim([0,1])
    title = "".join(("Approximation Ratio as a function of # of Qubits with Reptitions = ", str(repetitions)))
    title = "".join((title, " , Maxiter = ", str(maxiter)))
    title = "".join((title, " , and P = ", str(max_p)))
    ax.set_title(title)
    plt.plot(tmp, yValues)
    plt.ylabel('C/Cmin')
    plt.xlabel("# of Qubits")
    # plt.label(title)
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
