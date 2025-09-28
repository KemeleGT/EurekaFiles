#!/usr/bin/env python3
"""
Ready-to-use Fault Attack Automation Tool
Copy this file exactly as-is to your project directory
"""

import os
import json
import subprocess
import itertools
import pandas as pd
from datetime import datetime
import time
import logging
from typing import Dict, List, Any
import configparser
from pathlib import Path
import re

class FaultAttackAutomator:
    def __init__(self, base_config_path: str, output_dir: str = "experiment_results"):
        """Initialize the fault attack automator"""
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
        self.results = []
        
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
            
            # Parse the param string as a Python dict
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
        with open(config_path, 'w') as f:
            config.write(f)
        
        return str(config_path)
    
    def run_single_experiment(self, config_path: str, experiment_id: str) -> Dict[str, Any]:
        """Run a single experiment with given configuration"""
        self.logger.info(f"Running experiment {experiment_id}")
        
        start_time = time.time()
        
        try:
            # Run using the wrapper script
            log_file = self.output_dir / f"automation_tool_{experiment_id}.log"
            cmd = f"python cfg_wrapper.py {config_path} > {log_file} 2>&1"
            result = subprocess.run(cmd, shell=True, timeout=3600)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse the log file for results
            log_results = self._parse_log_file(str(log_file))
            
            experiment_result = {
                'experiment_id': experiment_id,
                'config_path': config_path,
                'log_file': str(log_file),
                'duration_seconds': duration,
                'return_code': result.returncode,
                'timestamp': datetime.now().isoformat(),
                **log_results
            }
            
            if result.returncode == 0:
                self.logger.info(f"Experiment {experiment_id} completed successfully")
            else:
                self.logger.error(f"Experiment {experiment_id} failed with return code {result.returncode}")
                
            return experiment_result
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Experiment {experiment_id} timed out")
            return {
                'experiment_id': experiment_id,
                'config_path': config_path,
                'duration_seconds': 3600,
                'return_code': -1,
                'error': 'Timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            return {
                'experiment_id': experiment_id,
                'config_path': config_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_log_file(self, log_file: str) -> Dict[str, Any]:
        """Parse the log file to extract results"""
        log_results = {}
        
        if not os.path.exists(log_file):
            return log_results
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract common metrics (customize based on your log format)
            
            # Look for accuracy patterns
            accuracy_patterns = [
                r'accuracy[:\s]+([0-9]+\.?[0-9]*)',
                r'acc[:\s]+([0-9]+\.?[0-9]*)',
                r'test accuracy[:\s]+([0-9]+\.?[0-9]*)'
            ]
            
            for pattern in accuracy_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        log_results['accuracy'] = float(matches[-1])  # Take last match
                        break
                    except:
                        pass
            
            # Look for error rate patterns
            error_patterns = [
                r'error rate[:\s]+([0-9]+\.?[0-9]*)',
                r'error[:\s]+([0-9]+\.?[0-9]*)',
                r'loss[:\s]+([0-9]+\.?[0-9]*)'
            ]
            
            for pattern in error_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        log_results['error_rate'] = float(matches[-1])
                        break
                    except:
                        pass
            
            # Look for other metrics
            if 'success' in content.lower():
                log_results['success'] = True
            if 'failed' in content.lower() or 'error' in content.lower():
                log_results['has_errors'] = True
                
        except Exception as e:
            self.logger.warning(f"Failed to parse log file {log_file}: {e}")
        
        return log_results
    
    def run_parameter_sweep(self, param_ranges: Dict[str, List[Any]]) -> pd.DataFrame:
        """Run a complete parameter sweep experiment"""
        param_combinations = self.define_parameter_space(param_ranges)
        
        self.logger.info(f"Starting parameter sweep with {len(param_combinations)} experiments")
        
        for i, params in enumerate(param_combinations):
            experiment_id = f"exp_{i+1:04d}"
            config_filename = f"config_{experiment_id}.cfg"
            
            # Create config file
            config_path = self.create_config_file(params, config_filename)
            
            # Add parameters to result
            result = params.copy()
            
            # Run experiment
            experiment_result = self.run_single_experiment(config_path, experiment_id)
            result.update(experiment_result)
            
            self.results.append(result)
            
            # Progress update
            progress = (i + 1) / len(param_combinations) * 100
            self.logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(param_combinations)})")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"experiment_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Also save as Excel if possible
        try:
            excel_file = self.output_dir / f"experiment_results_{timestamp}.xlsx"
            results_df.to_excel(excel_file, index=False)
            self.logger.info(f"Results saved to {excel_file}")
        except ImportError:
            self.logger.warning("openpyxl not available, skipping Excel export")
        
        self.logger.info(f"Results saved to {results_file}")
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a summary report of the experiments"""
        report_path = self.output_dir / "experiment_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Fault Attack Experiment Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(results_df)}\n\n")
            
            # Summary statistics
            if 'return_code' in results_df.columns:
                successful = len(results_df[results_df['return_code'] == 0])
                f.write(f"Successful Experiments: {successful}\n")
                f.write(f"Failed Experiments: {len(results_df) - successful}\n\n")
            
            # Parameter ranges
            f.write("Parameter Ranges:\n")
            for col in results_df.columns:
                if col not in ['experiment_id', 'config_path', 'duration_seconds', 
                             'return_code', 'timestamp', 'error', 'log_file']:
                    unique_values = results_df[col].unique()
                    f.write(f"  {col}: {list(unique_values)}\n")
            
            f.write("\n")
            
            # Results summary
            if 'accuracy' in results_df.columns:
                accuracy_data = results_df['accuracy'].dropna()
                if len(accuracy_data) > 0:
                    f.write(f"Accuracy Range: {accuracy_data.min():.3f} - {accuracy_data.max():.3f}\n")
                    f.write(f"Mean Accuracy: {accuracy_data.mean():.3f}\n")
            
            if 'error_rate' in results_df.columns:
                error_data = results_df['error_rate'].dropna()
                if len(error_data) > 0:
                    f.write(f"Error Rate Range: {error_data.min():.3f} - {error_data.max():.3f}\n")
                    f.write(f"Mean Error Rate: {error_data.mean():.3f}\n")
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)