"""
OrcaMet Forecast Engine — Core

Extracted from Point_Job_Certainty.py and adapted for Django.
Fetches multi-model ensemble forecasts from Open-Meteo, blends them
with geographic-aware weighting, and computes hourly risk scores.
"""

import logging
import math
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# ============================================================
# MULTI-MODEL CONFIGURATION
# ============================================================

MODELS_CONFIG = {
    "ukv": {
        "name": "Met Office UKV",
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "ukmo_uk_deterministic_2km"},
        "resolution_km": 2.0,
    },
    "ecmwf": {
        "name": "ECMWF HRES",
        "url": "https://api.open-meteo.com/v1/ecmwf",
        "params": {},
        "resolution_km": 9.0,
    },
    "icon_eu": {
        "name": "DWD ICON-EU",
        "url": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {},
        "resolution_km": 7.0,
    },
    "arpege": {
        "name": "Météo-France ARPEGE",
        "url": "https://api.open-meteo.com/v1/meteofrance",
        "params": {"models": "arpege_world"},
        "resolution_km": 10.0,
    },
}


# ============================================================
# GEOGRAPHIC-AWARE MODEL WEIGHTING
# ============================================================

def get_model_weights(lat: float, lon: float, exposure: str = "urban") -> dict:
    """Determine model weights based on geographic location and site exposure."""

    scotland = lat > 56.0
    northern_england = 53.5 < lat <= 56.0
    coastal = exposure == "coastal"
    highland = exposure == "highland"

    if highland or scotland:
        return {"ukv": 0.60, "ecmwf": 0.25, "icon_eu": 0.10, "arpege": 0.05}
    elif coastal:
        return {"ukv": 0.45, "ecmwf": 0.25, "arpege": 0.20, "icon_eu": 0.10}
    elif northern_england:
        return {"ukv": 0.40, "ecmwf": 0.30, "icon_eu": 0.20, "arpege": 0.10}
    else:
        return {"ukv": 0.35, "ecmwf": 0.35, "icon_eu": 0.20, "arpege": 0.10}


# ============================================================
# RISK MODEL
# ============================================================

def sigmoid(x: float) -> float:
    """Sigmoid activation for risk scoring."""
    return 1.0 / (1.0 + math.exp(-x))


def ramp(value: float, soft: float, hard: float, high_bad: bool = True) -> float:
    """Linear ramp between soft and hard thresholds."""
    if np.isnan(value):
        return np.nan
    if high_bad:
        if value <= soft:
            return 0.0
        if value >= hard:
            return 1.0
        return (value - soft) / (hard - soft)
    else:
        if value >= soft:
            return 0.0
        if value <= hard:
            return 1.0
        return (soft - value) / (soft - hard)


def calculate_hourly_risk(wind: float, gust: float, precip: float, temp: float,
                          thresholds: dict = None) -> float:
    """
    Calculate instantaneous hourly risk score (0-100%).

    Uses site-specific thresholds if provided, otherwise falls back to defaults.
    """
    if thresholds is None:
        thresholds = {
            "wind_mean_caution": 10.0, "wind_mean_cancel": 14.0,
            "gust_caution": 15.0, "gust_cancel": 20.0,
            "precip_caution": 0.7, "precip_cancel": 2.0,
            "temp_min_caution": 1.0, "temp_min_cancel": -2.0,
        }

    r = (
        0.30 * ramp(wind, thresholds["wind_mean_caution"], thresholds["wind_mean_cancel"], high_bad=True) +
        0.40 * ramp(gust, thresholds["gust_caution"], thresholds["gust_cancel"], high_bad=True) +
        0.20 * ramp(precip, thresholds["precip_caution"], thresholds["precip_cancel"], high_bad=True) +
        0.10 * ramp(temp, thresholds["temp_min_caution"], thresholds["temp_min_cancel"], high_bad=False)
    )

    if np.isnan(r):
        return np.nan

    prob = sigmoid(6.0 * (r - 0.45))
    return float(np.clip(prob * 100, 0.0, 100.0))


def get_recommendation(risk: float) -> str:
    """Convert risk score to recommendation string."""
    if np.isnan(risk):
        return "UNKNOWN"
    if risk < 20:
        return "GO"
    elif risk < 50:
        return "CAUTION"
    else:
        return "CANCEL"


# ============================================================
# API FETCHING
# ============================================================

def fetch_single_model(model_name: str, lat: float, lon: float,
                       start_date: str, end_date: str) -> dict:
    """Fetch hourly data from a single weather model via Open-Meteo."""

    config = MODELS_CONFIG[model_name]
    api_key = getattr(settings, "OPENMETEO_API_KEY", "")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_gusts_10m,precipitation,temperature_2m",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "start_date": start_date,
        "end_date": end_date,
        **config["params"],
    }
    if api_key:
        params["apikey"] = api_key

    resp = requests.get(config["url"], params=params, timeout=30)
    resp.raise_for_status()
    j = resp.json()

    h = j.get("hourly", {})
    if not h or "time" not in h:
        raise ValueError(f"No hourly data returned for {model_name}")

    return {
        "model": model_name,
        "time": h["time"],
        "wind_speed": h.get("wind_speed_10m", []),
        "wind_gusts": h.get("wind_gusts_10m", []),
        "precipitation": h.get("precipitation", []),
        "temperature": h.get("temperature_2m", []),
    }


def fetch_ensemble(lat: float, lon: float, exposure: str,
                   start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch multi-model ensemble for a location and blend into a single DataFrame.

    Returns DataFrame with columns: time, wind_speed, wind_gusts, precipitation,
    temperature, and _spread columns for each, plus n_models.
    """
    weights = get_model_weights(lat, lon, exposure)

    ensemble_data = {}
    successful_models = []

    for model_name, weight in weights.items():
        if model_name not in MODELS_CONFIG:
            continue
        try:
            data = fetch_single_model(model_name, lat, lon, start_date, end_date)
            ensemble_data[model_name] = {"weight": weight, "data": data}
            successful_models.append(model_name)
            logger.debug(f"  ✓ {MODELS_CONFIG[model_name]['name']}")
        except Exception as e:
            logger.warning(f"  ✗ {MODELS_CONFIG[model_name]['name']}: {e}")

        # Be polite to the API
        time.sleep(0.15)

    if not ensemble_data:
        raise ValueError(f"All models failed for ({lat:.4f}, {lon:.4f})")

    # Re-normalise weights to account for failed models
    total_weight = sum(d["weight"] for d in ensemble_data.values())
    for model_name in ensemble_data:
        ensemble_data[model_name]["weight"] /= total_weight

    return _create_weighted_ensemble(ensemble_data, successful_models)


def _create_weighted_ensemble(ensemble_data: dict, model_names: list) -> pd.DataFrame:
    """Blend multiple model outputs into a weighted ensemble DataFrame."""

    ref_data = list(ensemble_data.values())[0]["data"]
    times = pd.to_datetime(ref_data["time"], utc=True)
    n_times = len(times)

    ensemble_vars = {
        "wind_speed": np.zeros(n_times),
        "wind_gusts": np.zeros(n_times),
        "precipitation": np.zeros(n_times),
        "temperature": np.zeros(n_times),
    }
    raw_values = {var: [] for var in ensemble_vars}

    for model_name, model_info in ensemble_data.items():
        weight = model_info["weight"]
        data = model_info["data"]

        for var in ensemble_vars:
            values = data.get(var)
            if values is None:
                values = [np.nan] * n_times
            values = np.array(values)
            values = np.where(values is None, np.nan, values).astype(float)
            if len(values) == n_times:
                ensemble_vars[var] += weight * values
                raw_values[var].append(values)

    spread = {}
    for var, vals_list in raw_values.items():
        if len(vals_list) > 1:
            spread[f"{var}_spread"] = np.nanstd(vals_list, axis=0)
        else:
            spread[f"{var}_spread"] = np.zeros(n_times)

    df = pd.DataFrame({
        "time": times,
        **ensemble_vars,
        **spread,
        "n_models": len(model_names),
    })
    df.attrs["models_used"] = model_names
    return df
