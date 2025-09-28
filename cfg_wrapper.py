#!/usr/bin/env python3
"""
Ready-to-use wrapper script for cfg_model_parse.py
Handles config file switching automatically
"""

import sys
import os
import shutil
import subprocess

def run_with_config(config_file):
    """Run cfg_model_parse.py with a specific config file"""
    
    original_config = 'test_config_explore.cfg'
    backup_config = 'test_config_explore.cfg.backup'
    
    try:
        # Backup original config if it exists
        if os.path.exists(original_config):
            shutil.copy2(original_config, backup_config)
        
        # Copy the new config to the expected location
        shutil.copy2(config_file, original_config)
        
        # Run the original script
        result = subprocess.run([sys.executable, 'cfg_model_parse.py'], 
                              capture_output=True, text=True)
        
        # Print output to console and return result
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result
        
    finally:
        # Restore original config
        if os.path.exists(backup_config):
            shutil.move(backup_config, original_config)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cfg_wrapper.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found!")
        sys.exit(1)
    
    result = run_with_config(config_file)
    sys.exit(result.returncode)