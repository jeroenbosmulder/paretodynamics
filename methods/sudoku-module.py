import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import random as rd
import matplotlib.pyplot as plt
from functools import lru_cache

def get_onehotencoded(instance: np.ndarray) -> np.ndarray:
    """
    Converts the given Sudoku instance into one-hot encoded representation.
    
    Args:
        instance (np.ndarray): A 9x9 numpy array representing a Sudoku puzzle.
    
    Returns:
        instance_onehotencoded (np.ndarray): A one-hot encoded representation of the input Sudoku puzzle (81x10).
    """
    instance_asarray = instance.ravel()
    instance_onehotencoded = np.zeros((81, 9+1), dtype=float)
    instance_onehotencoded[np.arange(81),instance_asarray] = 1
    return(instance_onehotencoded)

def get_randomized(instance: np.ndarray) -> np.ndarray:
    """
    Randomizes the given one-hot encoded Sudoku instance.
    
    Args:
        instance (np.ndarray): A one-hot encoded representation of a Sudoku puzzle (81x10).
    
    Returns:
        instance_randomized (np.ndarray): A randomized one-hot encoded representation of the input Sudoku puzzle (81x9).
    """
    instance_randomized = instance   
    for key, entry in enumerate(instance_randomized):
        if entry[0] == 1:
            random_numbers = np.random.rand(9)  # Replace with NumPy's random float generation
            instance_randomized[key] = np.insert(random_numbers / sum(random_numbers), 0, 1)
    return instance_randomized[:, 1:]

@lru_cache(maxsize=None)
def get_constraints() -> list:
    """
    Generates row, column, and block constraints for a Sudoku puzzle.
    
    Returns:
        constraints (list): A list containing row, column, and block constraints for a Sudoku puzzle.
    """
    matrix = np.reshape(list(range(81)),(9,9))
    # Row constraints
    constraints = matrix.tolist()
    # Column constraints
    constraints += matrix.transpose().tolist()
    # Block constraints
    constraints += [(matrix[0:3,0:3]+i).ravel().tolist() 
                        for i in list(3*i+27*j 
                            for (i,j) in itertools.product(range(3),range(3))
                        )
                   ]
    return(constraints)

def get_constraints_by_reference(constraints: list = get_constraints()) -> defaultdict:
    """
    Maps constraint entries to their corresponding constraints.
    
    Args:
        constraints (list): A list containing row, column, and block constraints for a Sudoku puzzle.
    
    Returns:
        reference (defaultdict): A defaultdict that maps each entry to its corresponding constraints.
    """
    reference = defaultdict(list)
    for constraint in constraints:
        for entry in constraint:
            reference[entry].append(constraint)
    return(reference)

def get_entropy(instance: np.ndarray) -> float:
    """
    Calculates the entropy of the given one-hot encoded Sudoku instance.
    
    Args:
        instance (np.ndarray): A one-hot encoded representation of a Sudoku puzzle.
    
    Returns:
        entropy (float): The entropy of the input Sudoku instance.
    """
    entropy = 0
    for neuron in instance:
        entropy+=sum(-entry*np.log(entry) for entry in neuron if entry>0)
    return(entropy)

@lru_cache(maxsize=None)
def get_iterator() -> np.ndarray:
    """
    Generates a list of index pairs for combinations of numbers from 0 to 8.
    
    Returns:
        iterator (np.ndarray): A numpy array of shape (N, 2) containing index pairs.
    """
    return np.array(list(itertools.combinations(range(9), 2)))

def get_energy(puzzle_onehotencoded: np.ndarray, constraints: list = get_constraints(),
               iterator: np.ndarray = get_iterator()) -> int:
    """
    Calculates the energy of the given one-hot encoded Sudoku instance based on constraints.
    
    Args:
        puzzle_onehotencoded (np.ndarray): A one-hot encoded representation of a Sudoku puzzle.
        constraints (list): A list containing row, column, and block constraints for a Sudoku puzzle.
        iterator (np.ndarray): A precomputed numpy array of index pairs for combinations of numbers from 0 to 8.
    
    Returns:
        conflicts (int): The energy of the input Sudoku instance, which is the number of conflicts in the puzzle based on the constraints.
    """
    conflicts = 0
    squared = 0
    for constraint in constraints:
        entry = puzzle_onehotencoded[constraint]
        conflicts += np.sum(entry[iterator[:, 0]] * entry[iterator[:, 1]])
        squared += np.sum((entry[iterator[:, 0]] * entry[iterator[:, 1]])**2)
    return conflicts, squared

def get_saturation(instance: np.ndarray) -> float:
    """
    Calculates the saturation of the given Sudoku instance.
    
    Args:
        instance (np.ndarray): A one-hot encoded representation of a Sudoku puzzle.
    
    Returns:
        saturation (float): The saturation of the input Sudoku instance.
    """
    return(np.sum([(entry*entry) for entry in instance])/81)

def reshape(instance: str) -> np.ndarray:
    """
    Reshapes the given Sudoku instance string into a 9x9 numpy array.
    
    Args:
        instance (str): A string representing a Sudoku puzzle.
    
    Returns:
        reshaped_instance (np.ndarray): A 9x9 numpy array representation of the input Sudoku puzzle.
    """
    return(np.reshape([int(entry) for entry in instance],(9,9)))

def iterate(nodes, max_iterations):
    for idx in range(max_iterations):
        np.random.shuffle(nodes)
        for node in nodes:
            yield node

def simulated_annealing(puzzle: np.ndarray, max_iterations: int = 1000, T: float = 10, factor: float = 0.95) -> np.ndarray:
    """
    Solves the given Sudoku puzzle using a simulated annealing algorithm.

    Args:
        puzzle (np.ndarray): A 9x9 numpy array representing a Sudoku puzzle.
        max_iterations (int): Maximum number of iterations for the simulated annealing algorithm.
        T (float): Initial temperature for the simulated annealing algorithm.
        factor (float): Temperature reduction factor for the simulated annealing algorithm.

    Returns:
        instance (np.ndarray): A one-hot encoded representation of the solved Sudoku puzzle.
    """
    constraints_by_reference = get_constraints_by_reference()
    nodes = [key for key, value in enumerate(puzzle.ravel()) if value == 0]

    instance = get_randomized(get_onehotencoded(puzzle))

    entropies = [get_entropy(instance)]
    energy, squared = get_energy(instance)
    energies = [energy]
    squares = [squared]
    heat = [0]
    _heat = [0]
    joint_entropies = [entropies[-1]]
    temperatures = []
    empirical_temperatures = []

    for node in iterate(nodes, max_iterations):
        # Pick the neuron corresponding to the give node
        # This will be neuron that's get updated
        neuron_old = instance[node]
        # The new values for this neuron are determined by
        # its neighbouring neurons, that is the neurons to
        # which the current neuron is constrained
        neuron_tmp = sum(instance[entry] for
                     constraint in constraints_by_reference[node] for
                     entry in constraint if entry != node
                    )
        # The interaction by the neighbouring neurons is governed by T
        neuron_new = np.exp(-neuron_tmp / T)
        # The new neuron needs to be normalized
        neuron_new /= sum(neuron_new)
        # We can compute the change in entropy and energy by merely
        # looking "locally" at what has changed for the updated neuron
        delta_entropy = get_entropy([neuron_new])-get_entropy([neuron_old])
        energy_old = np.dot(neuron_old, neuron_tmp)
        energy_new = np.dot(neuron_new, neuron_tmp)
        delta_energy = energy_new - energy_old
        delta_squared = energy_new**2 - energy_old**2
        # And now we need to update the instance with the new neuron
        instance[node] = neuron_new
        entropies.append(entropies[-1]+delta_entropy)
        energies.append(energies[-1]+delta_energy)
        squares.append(squares[-1]+delta_squared)
        joint_entropy = entropies[-1]-energies[-1]/T
        joint_entropies.append(joint_entropy)
        if (delta_entropy)!=0:
            empirical_temperature = delta_energy / delta_entropy
        if abs(delta_entropy) < 0.001:
            T *= factor
        temperatures.append(T)
        empirical_temperatures.append(empirical_temperature)
        heat.append((squares[-1]-energies[-1]**2)/temperatures[-1]**2)
        _heat.append((squares[-1]-energies[-1]**2)/empirical_temperatures[-1]**2)
        if(T<0.1):
            break
                
    final_energy, final_squared = get_energy(np.round(instance, 1))
    print(final_energy)
    return instance, entropies, energies, joint_entropies, temperatures, empirical_temperatures, heat, _heat