"""
Utility functions for loading and managing configuration.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the smart grid fault detection system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.type' or 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config
    
    def save(self, output_path: str = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    @property
    def grid_config(self) -> Dict[str, Any]:
        """Get grid configuration."""
        return self.config.get('grid', {})
    
    @property
    def iot_config(self) -> Dict[str, Any]:
        """Get IoT configuration."""
        return self.config.get('iot', {})
    
    @property
    def detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.config.get('detection', {})


def get_config(config_path: str = None) -> Config:
    """
    Get configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Configuration loaded successfully!")
    print(f"\nFault types: {config.get('data.fault_types')}")
    print(f"Model type: {config.get('model.type')}")
    print(f"Batch size: {config.get('training.batch_size')}")
    print(f"Grid voltage: {config.get('grid.voltage_nominal')} V")
