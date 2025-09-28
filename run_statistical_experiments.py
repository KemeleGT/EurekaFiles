from statistical_fault_attack_automator import StatisticalFaultAttackAutomator

# Initialize
automator = StatisticalFaultAttackAutomator(
    base_config_path="test_config_explore.cfg",
    output_dir="statistical_results"
)

# Your parameter ranges
param_ranges = {
    'error_num': [50],
    'attack_layer': [5],
    'bit_num': [3],
    'distance_level': ['mantissa'],
    'num_of_runs': [3]  
}

# Run experiments
results = automator.run_parameter_sweep(param_ranges)
automator.generate_report(results)