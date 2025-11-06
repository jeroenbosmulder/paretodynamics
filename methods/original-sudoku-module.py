import numpy as np
import pandas as pd
import random

def get_constraints(N):
    square = np.arange(1, N**4 + 1).reshape(N**2, N**2)
    constraints = []
    divisors = [1, N, N**2]
    
    for d in divisors:
        for i in range(d):
            for j in range(N**2//d):
                constraints.append(square[(i*N**2//d):((i+1)*N**2//d),(j*d):((j+1)*d)].flatten())
                
    return np.array(constraints)


def get_weight_matrix(N):
    constraints = get_constraints(N)
    weight_matrix = np.zeros((N**4, N**4), dtype=int)
    
    for i in range(N**4):
        for j in range(N**4):
            weight_matrix[i, j] = np.sum(np.logical_or(constraints == i+1, constraints == j+1).sum(axis=1) == 2)
            
    return weight_matrix


def saturation_sudoku(values,N):
    return np.sum(values**2) / N**4


def deviation_sudoku(sudoku):
    values = np.apply_along_axis(lambda x: np.sum(x**2), 1, sudoku['values'])
    return np.var(values)


def energy(values, constraints):
    result = 0
    num_rows = constraints.shape[0]
    
    for i in range(num_rows):
        constraint = constraints[i]
        for j in range(len(constraint)):
            for k in range(j, len(constraint)):
                if k > j:
                    result += np.sum(values[constraint[j]-1] * values[constraint[k]-1])
                    
    return result

def energy_squared(values, constraints):
    result = 0
    num_rows = constraints.shape[0]
    
    for i in range(num_rows):
        constraint = constraints[i]
        for j in range(len(constraint)):
            for k in range(j, len(constraint)):
                if k > j:
                    result += np.sum((values[constraint[j]-1] * values[constraint[k]-1])**2)
                    
    return result


def entropy(values):
    return(-np.sum(np.ma.log(values)*values))


def rounded_sudoku(values, constraints):
    expectation = np.round(np.dot(values, np.arange(1, 10)))
    solution = np.array([[(1 if x == i else 0) for i in range(1, 10)] for x in expectation])
    energy_value = energy(solution, constraints)
    return {'Solution': solution, 'Energy': energy_value}


def logging_sudoku(values,constraints,T):
    E = energy(values,constraints)
    E2= energy_squared(values,constraints)
    S = entropy(values)
    J = S-E/T
    return(J,[E,S,T,J,E2])


def read_sudoku(file_name):
    sudoku = pd.read_csv(file_name, header=None, sep="\s+").values
    N = int(np.sqrt(sudoku.shape[0]))
    values = []
    fixed = []
    
    for i in range(N**2):
        for j in range(N**2):
            row = [0] * (N**2)
            if sudoku[i, j] != ".":
                row[int(sudoku[i, j]) - 1] = 1.0
                fixed.append(i * N**2 + j)
            values.append(row)
            
    return {'values': np.array(values), 'fixed': fixed, 'N': N, 'sigma': None}


def write_sudoku(sudoku, constraints):
    N = sudoku['N']
    solution = rounded_sudoku(sudoku, constraints)['Solution']
    output = []
    
    for i in range(N**2):
        row = []
        for j in range(N**2):
            row.append(np.argsort(-solution[i * N**2 + j])[0] + 1)
        output.append(row)
            
    return np.array(output)


def initial_energy(sudoku, constraints):
    values = sudoku['values']
    fixed = sudoku['fixed']
    N = sudoku['N']
    
    for i in set(range(N**4)) - set(fixed):
        values[i] = [1/N**2] * N**2
        
    sudoku['initialEnergy'] = energy(values, constraints)
    sudoku['initialEnergySquared'] = energy_squared(values, constraints)
    return sudoku


def initialize_sudoku(sudoku, constraints):
    sudoku = initial_energy(sudoku, constraints)
    values = sudoku['values']
    fixed = sudoku['fixed']
    
    for i in range(81):
        if i not in fixed:
            som = 0
            for j in range(9):
                values[i, j] = np.random.uniform() / 81 + 1/9
                som += values[i, j]
            for j in range(9):
                values[i, j] /= som
                
    sudoku['values'] = values
    sudoku['sigma'] = saturation_sudoku(values,81)
    return sudoku


def solve_sudoku(sudoku,factor,constraints,weightMatrix):
    result = []
    logging = []
    T = 10
    sigma = 0
    joint_entropy = float('-inf')
    
    while True:
        N = sudoku['N']
        fixed = sudoku['fixed']
        values = sudoku['values']
        volgorde = random.sample(range(N**4), N**4)        

        for i in volgorde:
            if i not in fixed:
                som = 0
                for k in range(9):
                    values[i, k] = 0
                    for j in range(81):
                        values[i, k] += weightMatrix[i, j] * values[j, k]
                    values[i, k] = np.exp(-values[i, k] / T)
                    som += values[i, k]
                for k in range(9):
                    values[i, k] /= som

                J, current_result = logging_sudoku(values,constraints,T)
                delta = J - joint_entropy
                joint_entropy = J
                result.append(current_result)
        
        sigma_new = saturation_sudoku(values,N)
        if abs(sigma_new - sigma) < 0.001:
            T *= factor
        if sigma_new > 0.99 or T < 0.01:
            break
        sigma = sigma_new
        
    rounded = rounded_sudoku(values, constraints)
    
    return {'result': result,
            'energy': round(rounded['Energy'], 5),
            'valid': np.sum(rounded['Solution']) == 81,
           }