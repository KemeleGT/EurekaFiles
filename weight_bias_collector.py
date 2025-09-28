#!/usr/bin/env python3
"""
Weight Data Collector for Bit-Flip Experiments
Drop this file in your project folder - no modifications needed
"""
import torch
import numpy as np
import json
import os
from datetime import datetime

class WeightDataCollector:
    def __init__(self):
        self.data = {}
        self.output_dir = "weight_bias_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def collect_model_data(self, model, experiment_params):
        """Collect weight and bias data from model"""
        exp_id = f"layer{experiment_params.get('attack_layer', 0)}_err{experiment_params.get('error_num', 0)}_{experiment_params.get('distance_level', 'none')}_bit{experiment_params.get('bit_num', 0)}"
        
        # Collect layer data
        layers_data = {}
        total_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_data = module.weight.data.cpu().numpy()
                
                layer_info = {
                    'layer_type': type(module).__name__,
                    'weight_shape': list(weight_data.shape),
                    'weight_mean': float(np.mean(weight_data)),
                    'weight_std': float(np.std(weight_data)),
                    'weight_min': float(np.min(weight_data)),
                    'weight_max': float(np.max(weight_data)),
                    'weight_l2_norm': float(np.linalg.norm(weight_data)),
                    'weight_num_zeros': int(np.sum(weight_data == 0)),
                    'weight_num_params': int(weight_data.size)
                }
                
                total_params += weight_data.size
                
                # Collect bias data if exists
                if hasattr(module, 'bias') and module.bias is not None:
                    bias_data = module.bias.data.cpu().numpy()
                    layer_info.update({
                        'bias_shape': list(bias_data.shape),
                        'bias_mean': float(np.mean(bias_data)),
                        'bias_std': float(np.std(bias_data)),
                        'bias_l2_norm': float(np.linalg.norm(bias_data)),
                        'bias_num_params': int(bias_data.size)
                    })
                    total_params += bias_data.size
                
                layers_data[name] = layer_info
        
        # Store complete experiment data
        self.data[exp_id] = {
            'experiment_id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_params': experiment_params,
            'total_parameters': total_params,
            'num_layers': len(layers_data),
            'layers': layers_data
        }
        
        # Save immediately
        self.save_data(exp_id)
        
        print(f"Weight data collected: {len(layers_data)} layers, {total_params:,} parameters")
        return exp_id
    
    def save_data(self, exp_id):
        """Save data for specific experiment"""
        if exp_id in self.data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/{exp_id}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.data[exp_id], f, indent=2)
            
            print(f"Weight data saved: {filename}")

# Global instance
collector = WeightDataCollector()

def collect_weights(model, attack_layer, error_num, distance_level, bit_num):
    """Simple function to collect weight data"""
    experiment_params = {
        'attack_layer': attack_layer,
        'error_num': error_num,
        'distance_level': distance_level,
        'bit_num': bit_num
    }
    return collector.collect_model_data(model, experiment_params)