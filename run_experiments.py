#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed version that handles encoding issues
Just copy this file and run: python run_experiments_fixed.py
"""

import os
import sys
import locale

# Fix encoding issues at the start
if sys.platform.startswith('win'):
    # Windows encoding fix
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

try:
    from fault_attack_automator import FaultAttackAutomator
except UnicodeDecodeError:
    print("Encoding error detected. Trying alternative import...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fault_attack_automator", "fault_attack_automator.py")
    fault_attack_automator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fault_attack_automator)
    FaultAttackAutomator = fault_attack_automator.FaultAttackAutomator

def safe_print(text):
    """Print function that handles encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))

def main():
    safe_print("=" * 60)
    safe_print("Fault Attack Automation Tool - Encoding Fixed")
    safe_print("=" * 60)
    
    # Check if required files exist
    required_files = [
        'test_config_explore.cfg',
        'cfg_model_parse.py',
        'cfg_wrapper.py',
        'fault_attack_automator.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        safe_print("ERROR: Missing required files:")
        for file in missing_files:
            safe_print(f"  - {file}")
        safe_print("\nPlease ensure all files are in the same directory.")
        sys.exit(1)
    
    safe_print("✓ All required files found")
    safe_print("")
    
    # Initialize automator with encoding handling
    try:
        automator = FaultAttackAutomator(
            base_config_path="test_config_explore.cfg",
            output_dir="experiment_results"
        )
        safe_print("✓ Automator initialized successfully")
    except Exception as e:
        safe_print(f"ERROR: Failed to initialize automator: {e}")
        sys.exit(1)
    
    # Define parameter space - CUSTOMIZE THIS FOR YOUR EXPERIMENTS
    safe_print("Setting up parameter space...")
    
    param_ranges = {
        # Example parameters - modify these for your research
        'error_num': [0],          # Test different error numbers
        'attack_layer': [1],          # Test different attack layers
        'bit_num': [1],                  # Test different bit numbers
        'num_of_runs': [3],                  # Single run for testing
        'distance_level' : ['mantissa', 'exponent']
    }
    
    # Calculate total experiments
    total_experiments = 1
    for param, values in param_ranges.items():
        total_experiments *= len(values)
    
    safe_print(f"Parameter ranges:")
    for param, values in param_ranges.items():
        safe_print(f"  {param}: {values}")
    safe_print(f"\nTotal experiments to run: {total_experiments}")
    safe_print("")
    
    # Confirm before running
    try:
        response = input("Do you want to proceed? (y/n): ").lower().strip()
    except:
        response = "y"  # Default to yes if input fails
        
    if response != 'y' and response != 'yes':
        safe_print("Experiment cancelled.")
        sys.exit(0)
    
    safe_print("\nStarting experiments...")
    safe_print("=" * 60)
    
    try:
        # Run the parameter sweep
        results_df = automator.run_parameter_sweep(param_ranges)
        
        # Generate report
        report_path = automator.generate_report(results_df)
        
        safe_print("\n" + "=" * 60)
        safe_print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        safe_print("=" * 60)
        
        # Summary
        successful = len(results_df[results_df['return_code'] == 0]) if 'return_code' in results_df.columns else len(results_df)
        failed = len(results_df) - successful
        
        safe_print(f"Total experiments: {len(results_df)}")
        safe_print(f"Successful: {successful}")
        safe_print(f"Failed: {failed}")
        safe_print("")
        safe_print("Output files:")
        safe_print(f"  Results: experiment_results/ folder")
        safe_print(f"  Report: {report_path}")
        safe_print(f"  Logs: experiment_results/automation.log")
        safe_print("")
        
        # Show results preview
        if len(results_df) > 0:
            safe_print("Results preview:")
            try:
                safe_print(str(results_df.head()))
            except:
                safe_print("Results saved successfully (preview unavailable due to encoding)")
        
        safe_print("\n✓ All done! Check the experiment_results/ folder for detailed output.")
        
    except KeyboardInterrupt:
        safe_print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\nERROR: Experiment failed: {e}")
        safe_print("Check experiment_results/automation.log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()