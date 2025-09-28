#!/usr/bin/env python3
"""
Multi-Run Statistical Fault Attack Automator - Ready to Use
Captures ALL runs and calculates proper statistics (mean, std, etc.)
"""

import os
import configparser
import subprocess
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from typing import Dict, List, Any
from pathlib import Path
import re

class StatisticalFaultAttackAutomator:
    """
    Enhanced automator that captures ALL runs and calculates statistics
    """
    
    def __init__(self, base_config_path: str, output_dir: str = "experiment_results"):
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'automation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load base configuration
        self.base_config = self._load_base_config()
        
        # Results storage - now with multi-run support
        self.detailed_results = []  # All individual runs
        self.statistical_results = []  # Statistical summaries per experiment
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file"""
        try:
            config = configparser.ConfigParser()
            config.read(self.base_config_path)
            return {section: dict(config.items(section)) for section in config.sections()}
        except Exception as e:
            self.logger.error(f"Failed to load base config: {e}")
            raise
    
    def define_parameter_space(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def create_config_file(self, params: Dict[str, Any], config_filename: str) -> str:
        """Create a configuration file with specified parameters"""
        config = configparser.ConfigParser()
        
        # Copy base config
        for section_name, section_data in self.base_config.items():
            config.add_section(section_name)
            for key, value in section_data.items():
                config.set(section_name, key, str(value))
        
        # Update parameters in the param section
        if 'test_param_original_test1' in config:
            param_str = config.get('test_param_original_test1', 'param')
            
            try:
                import ast
                param_dict = ast.literal_eval(param_str)
                
                # Update with new parameters
                for key, value in params.items():
                    if key in param_dict:
                        param_dict[key] = value
                
                # Convert back to string
                updated_param_str = str(param_dict)
                config.set('test_param_original_test1', 'param', updated_param_str)
                
            except Exception as e:
                self.logger.warning(f"Could not parse param string: {e}")
        
        # Write config file
        config_path = self.output_dir / config_filename
        with open(config_path, 'w', encoding='utf-8') as f:
            config.write(f)
        
        return str(config_path)
    
    def run_single_experiment_all_runs(self, config_path: str, experiment_id: str, 
                                     params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with ALL runs captured and statistics calculated
        """
        
        num_runs = params.get('num_of_runs', 1)
        
        self.logger.info(f"Running experiment {experiment_id} with {num_runs} runs")
        
        # Storage for all runs of this experiment
        all_run_results = []
        
        # Run multiple iterations
        for run_number in range(1, num_runs + 1):
            self.logger.info(f"  Run {run_number}/{num_runs} for {experiment_id}")
            
            run_start_time = time.time()
            
            try:
                # Run single iteration using cfg_wrapper
                log_file = self.output_dir / f"automation_tool_{experiment_id}_run{run_number}.log"
                cmd = f"python cfg_wrapper.py {config_path} > {log_file} 2>&1"
                result = subprocess.run(cmd, shell=True, timeout=3600)
                
                run_end_time = time.time()
                run_duration = run_end_time - run_start_time
                
                # Parse log file for this specific run
                run_results = self._parse_log_file(str(log_file))
                
                # Create detailed run record
                run_record = {
                    'experiment_id': experiment_id,
                    'run_number': run_number,
                    'run_duration_seconds': run_duration,
                    'return_code': result.returncode,
                    'log_file': str(log_file),
                    'timestamp': datetime.now().isoformat(),
                    **params,  # Include all experiment parameters
                    **run_results  # Include parsed results (accuracy, etc.)
                }
                
                all_run_results.append(run_record)
                
                # Log this run's results
                accuracy = run_results.get('accuracy', 0.0)
                self.logger.info(f"    Run {run_number} accuracy: {accuracy:.4f}")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"Run {run_number} of {experiment_id} timed out")
                
                timeout_record = {
                    'experiment_id': experiment_id,
                    'run_number': run_number,
                    'run_duration_seconds': 3600,
                    'return_code': -1,
                    'error': 'Timeout',
                    'timestamp': datetime.now().isoformat(),
                    'accuracy': 0.0,  # Default for failed runs
                    **params
                }
                all_run_results.append(timeout_record)
                
            except Exception as e:
                self.logger.error(f"Run {run_number} of {experiment_id} failed: {e}")
                
                error_record = {
                    'experiment_id': experiment_id,
                    'run_number': run_number,
                    'error': str(e),
                    'return_code': -1,
                    'timestamp': datetime.now().isoformat(),
                    'accuracy': 0.0,  # Default for failed runs
                    **params
                }
                all_run_results.append(error_record)
        
        # Store all individual run results
        self.detailed_results.extend(all_run_results)
        
        # Calculate statistics across all runs
        statistical_summary = self._calculate_run_statistics(all_run_results, experiment_id, params)
        
        # Store statistical summary
        self.statistical_results.append(statistical_summary)
        
        return statistical_summary
    
    def _calculate_run_statistics(self, all_run_results: List[Dict[str, Any]], 
                                experiment_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics across all runs of an experiment"""
        
        # Extract accuracies from all runs
        accuracies = [run.get('accuracy', 0.0) for run in all_run_results if run.get('accuracy') is not None]
        successful_runs = [run for run in all_run_results if run.get('return_code', -1) == 0]
        
        # Calculate basic statistics
        num_runs = len(all_run_results)
        num_successful = len(successful_runs)
        success_rate = num_successful / num_runs if num_runs > 0 else 0.0
        
        if accuracies:
            accuracy_mean = np.mean(accuracies)
            accuracy_std = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
            accuracy_min = np.min(accuracies)
            accuracy_max = np.max(accuracies)
            accuracy_median = np.median(accuracies)
            accuracy_range = accuracy_max - accuracy_min
        else:
            accuracy_mean = accuracy_std = accuracy_min = accuracy_max = accuracy_median = accuracy_range = 0.0
        
        # Calculate confidence interval if enough successful runs
        confidence_interval_95_lower = confidence_interval_95_upper = None
        if len(accuracies) > 2:
            try:
                from scipy import stats
                confidence_level = 0.95
                degrees_freedom = len(accuracies) - 1
                t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                margin_error = t_critical * (accuracy_std / np.sqrt(len(accuracies)))
                
                confidence_interval_95_lower = accuracy_mean - margin_error
                confidence_interval_95_upper = accuracy_mean + margin_error
            except ImportError:
                self.logger.warning("scipy not available, skipping confidence interval calculation")
        
        # Calculate total experiment duration
        total_duration = sum(run.get('run_duration_seconds', 0) for run in all_run_results)
        
        # Create statistical summary
        statistical_summary = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            
            # Experiment parameters
            **params,
            
            # Run statistics
            'num_runs_attempted': num_runs,
            'num_runs_successful': num_successful,
            'success_rate': success_rate,
            'total_duration_seconds': total_duration,
            
            # Accuracy statistics
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'accuracy_min': accuracy_min,
            'accuracy_max': accuracy_max,
            'accuracy_median': accuracy_median,
            'accuracy_range': accuracy_range,
            
            # Confidence interval
            'confidence_interval_95_lower': confidence_interval_95_lower,
            'confidence_interval_95_upper': confidence_interval_95_upper,
            
            # All individual accuracies (for reference)
            'all_accuracies': accuracies,
            
            # Coefficient of variation (std/mean)
            'coefficient_of_variation': (accuracy_std / accuracy_mean) if accuracy_mean > 0 else None
        }
        
        # Log statistical summary
        self.logger.info(f"Experiment {experiment_id} statistics:")
        self.logger.info(f"  Successful runs: {num_successful}/{num_runs} ({success_rate:.1%})")
        self.logger.info(f"  Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
        self.logger.info(f"  Range: [{accuracy_min:.4f}, {accuracy_max:.4f}]")
        if confidence_interval_95_lower is not None:
            self.logger.info(f"  95% CI: [{confidence_interval_95_lower:.4f}, {confidence_interval_95_upper:.4f}]")
        
        return statistical_summary
    
    def _parse_log_file(self, log_file: str) -> Dict[str, Any]:
        """Parse log file to extract results"""
        
        log_results = {}
        
        if not os.path.exists(log_file):
            return log_results
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract accuracy (customize patterns based on your log format)
            accuracy_patterns = [
                r'accuracy[:\s]+([0-9]+\.?[0-9]*)',
                r'acc[:\s]+([0-9]+\.?[0-9]*)',
                r'test accuracy[:\s]+([0-9]+\.?[0-9]*)',
                r'final accuracy[:\s]+([0-9]+\.?[0-9]*)'
            ]
            
            for pattern in accuracy_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        log_results['accuracy'] = float(matches[-1])  # Take last match
                        break
                    except:
                        pass
            
            # Extract other metrics if present
            if 'error' in content.lower() and 'rate' in content.lower():
                error_matches = re.findall(r'error rate[:\s]+([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
                if error_matches:
                    try:
                        log_results['error_rate'] = float(error_matches[-1])
                    except:
                        pass
                        
        except Exception as e:
            self.logger.warning(f"Failed to parse log file {log_file}: {e}")
        
        return log_results
    
    def run_parameter_sweep(self, param_ranges: Dict[str, List[Any]]) -> pd.DataFrame:
        """Run complete parameter sweep with multi-run statistics"""
        
        param_combinations = self.define_parameter_space(param_ranges)
        
        self.logger.info(f"Starting parameter sweep with {len(param_combinations)} experiments")
        
        for i, params in enumerate(param_combinations):
            experiment_id = f"exp_{i+1:04d}"
            config_filename = f"config_{experiment_id}.cfg"
            
            # Create config file
            config_path = self.create_config_file(params, config_filename)
            
            # Run experiment with all runs
            self.run_single_experiment_all_runs(config_path, experiment_id, params)
            
            # Progress update
            progress = (i + 1) / len(param_combinations) * 100
            self.logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(param_combinations)})")
        
        # Create results DataFrames
        detailed_df = pd.DataFrame(self.detailed_results)
        statistical_df = pd.DataFrame(self.statistical_results)
        
        # Save results
        self._save_results(detailed_df, statistical_df)
        
        return statistical_df
    
    def _save_results(self, detailed_df: pd.DataFrame, statistical_df: pd.DataFrame):
        """Save both detailed and statistical results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results (all individual runs)
        detailed_csv = self.output_dir / f"detailed_results_{timestamp}.csv"
        detailed_df.to_csv(detailed_csv, index=False)
        self.logger.info(f"Detailed results (all runs) saved to {detailed_csv}")
        
        # Save statistical summary (one row per experiment)
        statistical_csv = self.output_dir / f"statistical_results_{timestamp}.csv"
        statistical_df.to_csv(statistical_csv, index=False)
        self.logger.info(f"Statistical summary saved to {statistical_csv}")
        
        # Save Excel files if possible
        try:
            detailed_excel = self.output_dir / f"detailed_results_{timestamp}.xlsx"
            statistical_excel = self.output_dir / f"statistical_results_{timestamp}.xlsx"
            
            detailed_df.to_excel(detailed_excel, index=False)
            statistical_df.to_excel(statistical_excel, index=False)
            
            self.logger.info(f"Excel files saved: {detailed_excel}, {statistical_excel}")
        except ImportError:
            self.logger.warning("openpyxl not available, skipping Excel export")
    
    def generate_report(self, statistical_df: pd.DataFrame) -> str:
        """Generate comprehensive report"""
        
        report_path = self.output_dir / "experiment_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Multi-Run Statistical Fault Attack Experiment Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(statistical_df)}\n")
            
            if len(statistical_df) > 0:
                total_runs = statistical_df['num_runs_attempted'].sum()
                successful_runs = statistical_df['num_runs_successful'].sum()
                f.write(f"Total Individual Runs: {total_runs}\n")
                f.write(f"Successful Runs: {successful_runs} ({successful_runs/total_runs:.1%})\n\n")
                
                # Statistical summary
                f.write("Statistical Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean Accuracy (across experiments): {statistical_df['accuracy_mean'].mean():.4f}\n")
                f.write(f"Standard Deviation: {statistical_df['accuracy_mean'].std():.4f}\n")
                f.write(f"Min Accuracy: {statistical_df['accuracy_mean'].min():.4f}\n")
                f.write(f"Max Accuracy: {statistical_df['accuracy_mean'].max():.4f}\n\n")
                
                # Parameter ranges
                f.write("Parameter Ranges Tested:\n")
                f.write("-" * 30 + "\n")
                for col in statistical_df.columns:
                    if col in ['error_num', 'attack_layer', 'bit_num', 'distance_level', 'num_of_runs']:
                        unique_values = statistical_df[col].unique()
                        f.write(f"  {col}: {sorted(unique_values)}\n")
                
                f.write("\n")
                
                # Most reliable experiments (lowest std)
                if 'accuracy_std' in statistical_df.columns:
                    f.write("Most Consistent Experiments (lowest std):\n")
                    f.write("-" * 40 + "\n")
                    most_consistent = statistical_df.nsmallest(5, 'accuracy_std')
                    for _, row in most_consistent.iterrows():
                        f.write(f"  {row['experiment_id']}: accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}\n")
                    
                    f.write("\n")
                
                # Most vulnerable configurations
                f.write("Most Vulnerable Configurations:\n")
                f.write("-" * 35 + "\n")
                most_vulnerable = statistical_df.nsmallest(5, 'accuracy_mean')
                for _, row in most_vulnerable.iterrows():
                    f.write(f"  {row['experiment_id']}: accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}")
                    f.write(f" (layer={row.get('attack_layer', 'N/A')}, errors={row.get('error_num', 'N/A')})\n")
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)

def main():
    """Ready-to-use example"""
    
    print("🔄 Multi-Run Statistical Fault Attack Automation")
    print("=" * 60)
    
    # Initialize automator
    automator = StatisticalFaultAttackAutomator(
        base_config_path="test_config_explore.cfg",
        output_dir="statistical_results"
    )
    
    # Define parameter ranges - CUSTOMIZE THIS FOR YOUR RESEARCH
    param_ranges = {
        'error_num': [50, 100, 200],           # 3 values
        'attack_layer': [5, 10, 15],           # 3 values
        'bit_num': [3, 7],                     # 2 values
        'distance_level': ['mantissa'],        # 1 value
        'num_of_runs': [3]                     # 3 runs per experiment for statistics
    }
    
    # Calculate total experiments and runs
    total_experiments = 1
    for param, values in param_ranges.items():
        if param != 'num_of_runs':
            total_experiments *= len(values)
    
    num_runs = param_ranges['num_of_runs'][0]
    total_individual_runs = total_experiments * num_runs
    
    print(f"Experiment Design:")
    print(f"  Total Experiments: {total_experiments}")
    print(f"  Runs per Experiment: {num_runs}")
    print(f"  Total Individual Runs: {total_individual_runs}")
    print(f"  Estimated Time: {total_individual_runs * 2:.0f} minutes")
    print()
    
    # Confirm before running
    response = input("Do you want to proceed? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Experiment cancelled.")
        return
    
    print("\nStarting experiments...")
    
    # Run parameter sweep
    statistical_results = automator.run_parameter_sweep(param_ranges)
    
    # Generate report
    report_path = automator.generate_report(statistical_results)
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED!")
    print("=" * 60)
    print(f"Results saved in: statistical_results/")
    print(f"Files generated:")
    print(f"  📊 detailed_results_TIMESTAMP.csv - All individual runs")
    print(f"  📈 statistical_results_TIMESTAMP.csv - Statistical summaries")
    print(f"  📋 experiment_report.txt - Analysis report")
    print()
    
    # Show quick summary
    if len(statistical_results) > 0:
        print("Quick Summary:")
        print(f"  Mean accuracy across all experiments: {statistical_results['accuracy_mean'].mean():.4f}")
        print(f"  Most vulnerable: {statistical_results['accuracy_mean'].min():.4f}")
        print(f"  Least vulnerable: {statistical_results['accuracy_mean'].max():.4f}")
        print(f"  Average std within experiments: {statistical_results['accuracy_std'].mean():.4f}")

if __name__ == "__main__":
    main()