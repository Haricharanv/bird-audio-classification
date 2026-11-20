"""Configuration loader supporting hierarchical YAMLs, CLI dot overrides, and attribute access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml


class Config(dict):
    """Dictionary subclass supporting dot-notation attribute access and nested dicts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)
            elif isinstance(value, list):
                self[key] = [Config(item) if isinstance(item, dict) else item for item in value]

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to standard Python dict."""
        plain_dict: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                plain_dict[key] = value.to_dict()
            elif isinstance(value, list):
                plain_dict[key] = [item.to_dict() if isinstance(item, Config) else item for item in value]
            else:
                plain_dict[key] = value
        return plain_dict


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge source dictionary into target dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _parse_value(val_str: str) -> Any:
    """Parse string representation of value into appropriate Python type."""
    val_lower = val_str.strip().lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    if val_lower in ("none", "null"):
        return None

    # Check for integer
    try:
        return int(val_str)
    except ValueError:
        pass

    # Check for float
    try:
        return float(val_str)
    except ValueError:
        pass

    # Check for JSON/YAML collection (list or dict)
    if (val_str.startswith("[") and val_str.endswith("]")) or (
        val_str.startswith("{") and val_str.endswith("}")
    ):
        try:
            return json.loads(val_str.replace("'", '"'))
        except Exception:
            try:
                return yaml.safe_load(val_str)
            except Exception:
                pass

    return val_str


def parse_cli_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parse CLI list of 'dot.key=value' into a nested dictionary."""
    nested_overrides: Dict[str, Any] = {}
    for item in overrides:
        item = item.strip()
        if not item or "=" not in item:
            continue
        key_part, val_part = item.split("=", 1)
        keys = key_part.strip().split(".")
        parsed_val = _parse_value(val_part.strip())

        curr = nested_overrides
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = parsed_val
    return nested_overrides


def load_config(
    config_path: Union[str, Path],
    cli_overrides: Union[List[str], None] = None,
) -> Config:
    """Load YAML config file, recursively resolving '_base_' references and applying CLI overrides.
    
    Args:
        config_path: Path to YAML config file.
        cli_overrides: Optional list of CLI strings like ['model.name=resnet34', 'debug=true'].
        
    Returns:
        Config object with dot access.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}

    # Handle recursive base configuration
    if "_base_" in cfg_dict:
        base_rel_path = cfg_dict.pop("_base_")
        # Base path can be relative to current config or workspace root
        base_path = (path.parent / base_rel_path).resolve()
        if not base_path.is_file():
            base_path = Path(base_rel_path).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config file not found: {base_rel_path} (resolved {base_path})")

        base_cfg = load_config(base_path)
        base_dict = base_cfg.to_dict()
        _deep_merge(base_dict, cfg_dict)
        cfg_dict = base_dict

    # Apply CLI dot overrides
    if cli_overrides:
        parsed_overrides = parse_cli_overrides(cli_overrides)
        _deep_merge(cfg_dict, parsed_overrides)

    return Config(cfg_dict)


def save_config(config: Union[Config, Dict[str, Any]], save_path: Union[str, Path]) -> None:
    """Save Config or dict to YAML file.
    
    Args:
        config: Config or dict instance.
        save_path: Destination path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dict_to_save = config.to_dict() if isinstance(config, Config) else config
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict_to_save, f, default_flow_style=False, sort_keys=False)

