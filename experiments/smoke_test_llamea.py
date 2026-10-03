import math
from types import SimpleNamespace

import numpy as np


class OverBudgetException(Exception):
    """The generated algorithm tried to call f() more times than its budget"""

def evaluate_simple(solution, explogger=None):
    code_str = solution.code
    alg_name = solution.name
    
    def f(x):
        if f.call_count >= 100:
            raise OverBudgetException("Budget exceeded.")
        f.call_count += 1
        return (x - 2.0) ** 2

    f.call_count = 0
    f.bounds = SimpleNamespace(lb=np.array([-5.0]), ub=np.array([5.0]))

    safe_globals = {
        "OverBudgetException": OverBudgetException,
        "math": math,
        "np": np,
    }
    local_env = {}
    try:
        exec(code_str, safe_globals, local_env)
    except Exception as e:
        solution.set_scores(0, feedback=f"Runtime/Syntax error: {e}")
        return solution

    try:
        AlgorithmClass = local_env[alg_name]
        algo = AlgorithmClass(budget=100, dim=1)
    except Exception as e:
        solution.set_scores(0, feedback=f"Instantiation error: {e}")
        return solution

    best_f = math.inf
    try:
        best_f, best_x = algo(f)
    except OverBudgetException:
        best_f = 9999
    except Exception as e:
        solution.set_scores(0, feedback=f"Runtime error during optimisation: {e}")
        return solution

    best_f = float(np.asarray(best_f).item())
    score = 1.0 / (1.0 + best_f)
    feedback_str = f"Algorithm {alg_name} got score={score:.4f} (bigger is better)."
    solution.set_scores(score, feedback_str)
    return solution


def main():
    from llamea import LLaMEA, Ollama_LLM

    llm = Ollama_LLM("qwen2.5-coder:1.5b") 

    role_prompt = "You are an AI that generates Python optimization code."
    task_prompt = """\
Create a Python class named Optimize, with __init__(self, budget, dim),
and a __call__(self, func) method that tries to find x in [-5, 5] that minimizes func(x).
The function should return the tuple best_f, best_x (value best found and the location).
Implement an efficient algorithm for this task.
"""

    es = LLaMEA(
        f=evaluate_simple,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        experiment_name="smoke-test",
        elitism=True,
        budget=3,
        eval_timeout=30, 
    )

    best_solution = es.run()
    print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness}")
    print(f"Generated code:\n{best_solution.code}")
    print(f"Additional feedback: {best_solution.feedback}")


if __name__ == "__main__":
    main()

 
