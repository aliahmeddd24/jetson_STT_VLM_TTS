"""YAML config with environment overrides.

Override any leaf with G1V__SECTION__KEY=value (double underscores separate
levels). Values are parsed as YAML, so G1V__VAD__THRESHOLD=0.6 gives a float.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


class Cfg(dict):
    """dict with attribute access for nested sections."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _set_path(d: Dict[str, Any], path: list, value: Any) -> None:
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str, overrides: Optional[Dict[str, Any]] = None,
         env: Optional[Dict[str, str]] = None, prefix: str = "G1V__") -> Cfg:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    env = os.environ if env is None else env
    for key, raw in env.items():
        if not key.startswith(prefix):
            continue
        parts = [p.lower() for p in key[len(prefix):].split("__") if p]
        if parts:
            _set_path(data, parts, yaml.safe_load(raw))
    if overrides:
        data = deep_merge(data, overrides)
    return _wrap(data)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj
