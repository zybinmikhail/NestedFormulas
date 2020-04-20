import time

import nested_formula
import auxiliary_functions
import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def print_sharps():
    print("\n################################################################################")


def print_results(symbolic_mse_list, number_of_tested_formulas, recovery_threshold=1e-5):
    print_sharps()
    print("MSEs between parameters:")
    print(symbolic_mse_list)
    symbolic_mse_list = np.array(symbolic_mse_list)
    number_of_small_errors = (symbolic_mse_list < recovery_threshold).sum()
    print(f"For {number_of_small_errors} formulas out of {number_of_tested_formulas} ", end="")
    print(f"the error is less than {recovery_threshold}.")


def print_time(cnt, cnt_iteration):
    time_from_start = time.perf_counter() - cnt
    time_iteration = time.perf_counter() - cnt_iteration
    min_from_start, sec_from_start = divmod(time_from_start, 60)
    min_iteration, sec_iteration = divmod(time_iteration, 60)
    print(
        f"{auxiliary_functions.remove_zero_minutes(min_from_start)}{sec_from_start :.0f} seconds passed from the start, ",
        end="")
    print(f"the iteration took {auxiliary_functions.remove_zero_minutes(min_iteration)}{sec_iteration :.0f} seconds")


def get_params(regressor, n_variables):
    lambdas = [regressor.get_lambda(i).item() for i in range(n_variables)]
    powers = [regressor.get_power(i).item() for i in range(n_variables)]
    bias_term = [regressor.last_subformula.lambda_0.item()]
    obtained_params = np.array(lambdas + powers + bias_term)
    return obtained_params


def generate_data(n_variables=3, m_samples=1000, min_power=1, max_power=2, divide_powers_by=1):
    X = torch.rand(m_samples, n_variables)
    b = torch.randn(1)
    coeffs = torch.randn((n_variables, 1))
    powers = torch.randint(min_power, max_power, (1, n_variables)) / float(divide_powers_by)
    y = X ** powers @ coeffs + b
    true_params = np.array(coeffs.view(-1, ).tolist() + powers.view(-1, ).tolist() + [b.item()])
    return coeffs, powers, b, X, y, true_params


def print_ground_truth(coeffs, powers, b):
    formula = []
    powers = powers.view(-1, )
    for i in range(len(coeffs)):
        new_term = [round(coeffs[i].item(), 3), "x_{", i + 1, "}^{", round(powers[i].item(), 3), "}"]
        if new_term[0] > 0 and i > 0:
            formula.append('+')
        formula.extend(list(map(str, new_term)))
    if b > 0:
        formula.append("+")
    formula.append(str(round(b.item(), 3)))
    auxiliary_functions.PrintFormula("".join(formula))


def explore(n_variables=3, m_samples=1000, min_power=1, max_power=1, number_of_tested_formulas=10,
            recovery_threshold=1e-4,
            divide_powers_by=1):
    cnt = time.perf_counter()
    symbolic_mse_list = []
    n_recoveries = 0
    for i in range(number_of_tested_formulas):
        print(f"\n\n----------------------------Exploring new formula #{i + 1}----------------------------")
        coeffs, powers, b, X, y, true_params = generate_data(n_variables, m_samples, min_power, max_power + 1,
                                                             divide_powers_by)
        cnt_iteration = time.perf_counter()
        regressor, _ = nested_formula.LearnFormula(X, y, optimizer_for_formula=torch.optim.Rprop, n_init=4)
        print_time(cnt, cnt_iteration)
        print("ground truth and obtained formula")
        print_ground_truth(coeffs, powers, b)
        auxiliary_functions.PrintFormula(regressor)

        obtained_params = get_params(regressor, n_variables)
        symbolic_mse = mean_squared_error(true_params, obtained_params)
        symbolic_mse_list.append(symbolic_mse)
        print(f"MSE between formula parameters is {symbolic_mse}")
        if symbolic_mse < recovery_threshold:
            print("EXACT RECOVERY")
            n_recoveries += 1
        else:
            print("FAILURE")

    print_results(symbolic_mse_list, number_of_tested_formulas)
    return n_recoveries
