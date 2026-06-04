"""
config_loader.py — Configuration management for JARVIS
Loads and validates configuration from config.yaml with sensible defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class ConfigLoader:
    """Load and manage JARVIS configuration."""
    
    DEFAULT_CONFIG = {
        "app": {"name": "JARVIS", "version": "2.0", "workspace": "~/JarvisData"},
        "llm": {
            "provider": "ollama",
            "default_model": "qwen3:8b",
            "deep_model": "qwen3:14b",
            "ollama_url": "http://localhost:11434"
        },
        "memory": {
            "enabled": True,
            "path": "./db",
            "relevance_threshold": 0.35,
            "max_entries": 10000
        },
        "security": {
            "sandbox_path": "~/JarvisData",
            "require_confirmation_for_high_risk": True,
            "allow_list": [
                "read_file", "list_dir", "search_files",
                "system_info", "memory_search",
                "open_app", "open_url", "search_web"
            ],
            "block_list": [],
            "max_file_size": 512000
        },
        "browser": {"enabled": True, "headless": False, "timeout": 30000, "max_tabs": 5},
        "voice": {"enabled": True, "wake_words": ["hey jarvis", "jarvis"], "tts_rate": 175, "tts_volume": 0.92},
        "audit": {"enabled": True, "log_dir": "./logs", "retention_days": 90},
        "plugins": {"enabled": True, "plugin_dir": "./plugins", "auto_load": True}
    }
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load config from file, or use defaults."""
        if not _YAML_AVAILABLE:
            return self.DEFAULT_CONFIG.copy()
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                return self._merge_configs(self.DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, defaults: dict, user_config: dict) -> dict:
        """Recursively merge user config into defaults."""
        result = defaults.copy()
        for key, value in user_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _save_config(self, config: dict) -> bool:
        """Save config to file."""
        if not _YAML_AVAILABLE:
            return False
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-notation path."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set a config value by dot-notation path."""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        return self._save_config(self.config)
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration."""
        errors = []
        
        if "llm" not in self.config:
            errors.append("Missing 'llm' section")
        
        llm_cfg = self.get("llm", {})
        if llm_cfg.get("provider") == "ollama":
            url = llm_cfg.get("ollama_url")
            if not url:
                errors.append("Ollama URL not configured")
        
        sec_cfg = self.get("security", {})
        if not sec_cfg.get("sandbox_path"):
            errors.append("Security sandbox path not configured")
        
        if self.get("plugins.enabled"):
            plugin_dir = Path(self.get("plugins.plugin_dir", "./plugins"))
            plugin_dir.mkdir(exist_ok=True)
        
        return len(errors) == 0, errors
    
    def print_config(self) -> None:
        """Print the current config."""
        import json
        print(json.dumps(self.config, indent=2))
