"""
Configuration Parser Module
============================
Safely loads YAML configuration files into Python dictionaries
with support for nested key access via dot notation.
"""

import os
import yaml
from typing import Any, Optional


class ConfigParser:
    """Loads and provides access to YAML configuration values.

    Supports nested key access via dot notation (e.g., 'early_stopping.patience').
    All values are driven by the config file — no hard-coded defaults.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the ConfigParser by loading a YAML file.

        Args:
            config_path: Absolute or relative path to the YAML config file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file contains invalid YAML.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config: dict = yaml.safe_load(f)

        if self._config is None:
            self._config = {}

    @property
    def config(self) -> dict:
        """Return the full configuration dictionary."""
        return self._config

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'early_stopping.patience').
            default: Value to return if the key is not found.

        Returns:
            The configuration value, or default if not found.

        Examples:
            >>> cfg.get('batch_size')
            32
            >>> cfg.get('early_stopping.patience')
            5
            >>> cfg.get('paths.data_dir')
            './data'
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, key: str) -> str:
        """Get a path value and ensure the directory exists.

        Args:
            key: Dot-separated key path to a directory config value.

        Returns:
            The resolved path string.

        Raises:
            ValueError: If the key does not exist in the config.
        """
        path = self.get(key)
        if path is None:
            raise ValueError(f"Path key '{key}' not found in configuration.")

        os.makedirs(path, exist_ok=True)
        return path

    def __repr__(self) -> str:
        return f"ConfigParser(keys={list(self._config.keys())})"
