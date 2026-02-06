"""Query the HFT alpha simulation registry."""

import json
from pathlib import Path
from typing import Any
import pandas as pd


DATA_ROOT = Path("/home/labcmap/saad.souilmi/dev_cpp/qr/data")


def load_registry(ticker: str) -> dict:
    """Load the registry for a given ticker."""
    path = DATA_ROOT / "results" / ticker / "hft_alpha_results" / "registry.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _match(config: dict, query: dict) -> bool:
    """Check if config matches a partial query (nested)."""
    for key, val in query.items():
        if key not in config:
            return False
        if isinstance(val, dict):
            if not isinstance(config[key], dict) or not _match(config[key], val):
                return False
        else:
            if config[key] != val:
                return False
    return True


def search(ticker: str, **query) -> pd.DataFrame:
    """Search registry by partial config match.

    Examples:
        # Find all runs with steepness=20
        search("PFE", race={"steepness": 20.0})

        # Find all runs with kappa=1.0 and no impact
        search("PFE", impact="no_impact", ou={"kappa": 1.0})

        # Find all runs
        search("PFE")
    """
    registry = load_registry(ticker)
    results_dir = DATA_ROOT / "results" / ticker / "hft_alpha_results"

    rows = []
    for hash_id, entry in registry.items():
        config = entry.get("config", {})
        if _match(config, query):
            parquet_path = results_dir / f"{hash_id}.parquet"
            rows.append({
                "hash": hash_id,
                "file": str(parquet_path),
                "exists": parquet_path.exists(),
                "seed_used": entry.get("seed_used"),
                "timestamp": entry.get("timestamp"),
                **_flatten(config),
            })

    return pd.DataFrame(rows)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict: {"race": {"steepness": 8}} → {"race.steepness": 8}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def load_result(ticker: str, hash_id: str) -> pd.DataFrame:
    """Load a parquet result by hash."""
    path = DATA_ROOT / "results" / ticker / "hft_alpha_results" / f"{hash_id}.parquet"
    return pd.read_parquet(path)
