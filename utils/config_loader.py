"""
Configuration Loader for Neurogen v1.1

This module provides utilities to load and validate configuration files.
Supports YAML and Python dict configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return default config.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    
    Example:
        >>> config = load_config("configs/default.yaml")
        >>> print(config['training']['learning_rate'])
        0.01
    """
    if config_path is None:
        # Use default config path
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "configs", 
            "default.yaml"
        )
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary has required fields.
    
    Args:
        config: Configuration dictionary to validate
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['network', 'training', 'dataset', 'seed', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate network config
    if 'layer_sizes' not in config['network']:
        raise ValueError("network.layer_sizes is required")
    
    if len(config['network']['layer_sizes']) < 2:
        raise ValueError("network.layer_sizes must have at least 2 layers (input and output)")
    
    # Validate training config
    required_training = ['iterations', 'learning_rate']
    for field in required_training:
        if field not in config['training']:
            raise ValueError(f"training.{field} is required")
    
    # Validate seed config
    if 'random_seed' not in config['seed']:
        raise ValueError("seed.random_seed is required")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to config value (e.g., "training.learning_rate")
        default: Default value if key not found
    
    Returns:
        Config value or default
    
    Example:
        >>> config = load_config()
        >>> lr = get_config_value(config, "training.learning_rate", 0.01)
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save the config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
