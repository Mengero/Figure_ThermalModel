from arm_solver_utils import prepare_system, solve_system, postprocess_results

def main():
    system_data = prepare_system()
    solution_data = solve_system(system_data)
    postprocess_results(solution_data)

if __name__ == "__main__":
    main()