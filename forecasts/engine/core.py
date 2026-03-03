"""
OrcaMet Forecast Engine — Core

Extracted from Point_Job_Certainty.py and adapted for Django.
Fetches multi-model ensemble forecasts from Open-Meteo, blends them
with geographic-aware weighting, and computes hourly risk scores.

Models:
    UKV           – Met Office 2 km, UK mainland (always available for UK sites)
    ICON-D2       – DWD 2.2 km, Central Europe (SE England / East Anglia only)
    AROME France  – Météo-France 2.5 km, covers southern UK + Channel
    HARMONIE Europe – KNMI 5.5 km, full UK + Ireland + Northern Europe
    ARPEGE Europe – Météo-France 11 km, Europe-wide synoptic backbone
    ECMWF HRES    – 9 km global, best medium-range
    ICON-EU       – DWD 7 km, Europe-wide
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
    # --- High-resolution regional models (domain-limited) ---
    "ukv": {
        "name": "Met Office UKV",
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "ukmo_uk_deterministic_2km"},
        "resolution_km": 2.0,
        "forecast_days": 5,
    },
    "icon_d2": {
        "name": "DWD ICON-D2",
        "url": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {"models": "icon_d2"},
        "resolution_km": 2.2,
        "forecast_days": 2,
    },
    "arome_france": {
        "name": "Météo-France AROME",
        "url": "https://api.open-meteo.com/v1/meteofrance",
        "params": {"models": "arome_france"},
        "resolution_km": 2.5,
        "forecast_days": 2,
    },
    "harmonie_europe": {
        "name": "KNMI HARMONIE AROME Europe",
        "url": "https://api.open-meteo.com/v1/knmi",
        "params": {"models": "knmi_harmonie_arome_europe"},
        "resolution_km": 5.5,
        "forecast_days": 2,
    },
    # --- Synoptic / Europe-wide models (always available for UK) ---
    "arpege_europe": {
        "name": "Météo-France ARPEGE Europe",
        "url": "https://api.open-meteo.com/v1/meteofrance",
        "params": {"models": "arpege_europe"},
        "resolution_km": 11.0,
        "forecast_days": 4,
    },
    "ecmwf": {
        "name": "ECMWF IFS",
        "url": "https://api.open-meteo.com/v1/ecmwf",
        "params": {},
        "resolution_km": 9.0,
        "forecast_days": 10,
    },
    "icon_eu": {
        "name": "DWD ICON-EU",
        "url": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {"models": "icon_eu"},
        "resolution_km": 7.0,
        "forecast_days": 5,
    },
}

# ============================================================
# GEOGRAPHIC DOMAIN DEFINITIONS
# ============================================================
#
# Approximate bounding boxes for domain-limited models.
# These are conservative estimates — better to miss a marginal
# grid cell than to request data and get interpolated junk from
# outside the native domain.
#
# UKV: always eligible if the site is roughly within the UK
# ICON-D2: Central Europe — clips off Scotland, Wales, SW England
# AROME France: Southern UK, Channel, France, Iberia
# HARMONIE Europe: Full UK + Ireland + Scandinavia + Central Europe

DOMAIN_BOUNDS = {
    "ukv": {
        "lat_min": 49.5, "lat_max": 61.0,
        "lon_min": -12.0, "lon_max": 3.0,
    },
    "icon_d2": {
        # Conservative: only SE/E England and the near-continent
        "lat_min": 49.0, "lat_max": 55.5,
        "lon_min": -3.0, "lon_max": 16.0,
    },
    "arome_france": {
        # Covers southern England, Channel, France, NE Spain
        "lat_min": 38.0, "lat_max": 55.5,
        "lon_min": -8.0, "lon_max": 12.0,
    },
    "harmonie_europe": {
        # Very wide: Iceland to Central Europe, full UK coverage
        "lat_min": 45.0, "lat_max": 72.0,
        "lon_min": -30.0, "lon_max": 30.0,
    },
    # These three have Europe-wide or global coverage — always eligible
    "arpege_europe": None,  # Always available
    "ecmwf": None,          # Always available
    "icon_eu": None,         # Always available
}


def is_in_domain(model_name: str, lat: float, lon: float) -> bool:
    """Check whether a lat/lon point falls within a model's native domain."""
    bounds = DOMAIN_BOUNDS.get(model_name)
    if bounds is None:
        return True  # Global / Europe-wide models are always eligible
    return (
        bounds["lat_min"] <= lat <= bounds["lat_max"] and
        bounds["lon_min"] <= lon <= bounds["lon_max"]
    )


def get_eligible_models(lat: float, lon: float) -> list:
    """Return list of model names whose domains cover the given location."""
    return [m for m in MODELS_CONFIG if is_in_domain(m, lat, lon)]


# ============================================================
# GEOGRAPHIC-AWARE MODEL WEIGHTING
# ============================================================

# Base weighting tiers for UK regions.
# These represent the *ideal* weights when all models are available.
# If a model is not in domain, it is excluded and weights are re-normalised.

WEIGHT_PROFILES = {
    "scotland_highland": {
        # UKV dominates — knows the orography. HARMONIE adds value.
        # ICON-D2 and AROME almost certainly out of domain here.
        "ukv": 0.40, "harmonie_europe": 0.20, "ecmwf": 0.20,
        "icon_eu": 0.10, "arpege_europe": 0.10,
        "icon_d2": 0.00, "arome_france": 0.00,
    },
    "northern_england": {
        # UKV still strong. HARMONIE useful. AROME may clip in.
        "ukv": 0.30, "harmonie_europe": 0.20, "ecmwf": 0.20,
        "icon_eu": 0.10, "arpege_europe": 0.05,
        "arome_france": 0.10, "icon_d2": 0.05,
    },
    "south_east_england": {
        # All models available — spread the load, favouring high-res
        "ukv": 0.20, "icon_d2": 0.20, "arome_france": 0.15,
        "harmonie_europe": 0.15, "ecmwf": 0.15,
        "icon_eu": 0.10, "arpege_europe": 0.05,
    },
    "south_west_england": {
        # Atlantic influence — ARPEGE good for synoptic tracking.
        # ICON-D2 likely out of domain. AROME covers this.
        "ukv": 0.25, "arome_france": 0.20, "harmonie_europe": 0.15,
        "ecmwf": 0.15, "arpege_europe": 0.10, "icon_eu": 0.10,
        "icon_d2": 0.05,
    },
    "wales_irish_sea": {
        # Atlantic-exposed, UKV strong for Welsh valleys.
        # ICON-D2 out of domain. AROME marginal.
        "ukv": 0.30, "harmonie_europe": 0.20, "ecmwf": 0.15,
        "arpege_europe": 0.15, "icon_eu": 0.10,
        "arome_france": 0.10, "icon_d2": 0.00,
    },
    "coastal_channel": {
        # English Channel — AROME excels here, ICON-D2 available
        "arome_france": 0.25, "ukv": 0.20, "icon_d2": 0.15,
        "harmonie_europe": 0.15, "ecmwf": 0.10,
        "arpege_europe": 0.10, "icon_eu": 0.05,
    },
    "default_uk": {
        # Midlands / generic UK — balanced approach
        "ukv": 0.25, "harmonie_europe": 0.20, "ecmwf": 0.15,
        "arome_france": 0.10, "icon_d2": 0.10,
        "icon_eu": 0.10, "arpege_europe": 0.10,
    },
}


def _classify_region(lat: float, lon: float, exposure: str) -> str:
    """Classify a UK lat/lon into a weighting region."""
    # Channel coast special case
    if exposure == "coastal" and lat < 51.5 and lon > -5.0:
        return "coastal_channel"

    # Scotland and Highlands
    if lat > 56.0 or exposure == "highland":
        return "scotland_highland"

    # Northern England
    if lat > 53.5:
        return "northern_england"

    # Wales and Irish Sea coast
    if lon < -3.0:
        return "wales_irish_sea"

    # South East England
    if lon > -1.5 and lat < 53.5:
        return "south_east_england"

    # South West England
    if lon <= -1.5 and lat < 53.5:
        return "south_west_england"

    return "default_uk"


def get_model_weights(lat: float, lon: float, exposure: str = "urban") -> dict:
    """
    Determine model weights based on geographic location, site exposure,
    and model domain eligibility.

    Returns dict of {model_name: weight} for eligible models only.
    Weights are normalised to sum to 1.0.
    """
    region = _classify_region(lat, lon, exposure)
    profile = WEIGHT_PROFILES.get(region, WEIGHT_PROFILES["default_uk"]).copy()

    # Remove models that are outside their native domain for this location
    eligible = get_eligible_models(lat, lon)
    weights = {m: w for m, w in profile.items() if m in eligible and w > 0}

    if not weights:
        # Fallback: at least use the always-available models
        weights = {"ecmwf": 0.40, "icon_eu": 0.30, "arpege_europe": 0.30}

    # Re-normalise so weights sum to 1.0
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


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
        return 0.0

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

    Automatically determines which models are eligible based on geographic
    domain coverage, assigns region-aware weights, and blends into a single
    forecast with uncertainty spread metrics.

    Returns DataFrame with columns: time, wind_speed, wind_gusts, precipitation,
    temperature, and _spread columns for each, plus n_models and models_used.
    """
    weights = get_model_weights(lat, lon, exposure)

    logger.info(
        f"Ensemble for ({lat:.4f}, {lon:.4f}) [{exposure}]: "
        f"{len(weights)} models — {', '.join(f'{m}={w:.0%}' for m, w in weights.items())}"
    )

    ensemble_data = {}
    successful_models = []

    for model_name, weight in weights.items():
        if model_name not in MODELS_CONFIG:
            continue
        try:
            data = fetch_single_model(model_name, lat, lon, start_date, end_date)
            ensemble_data[model_name] = {"weight": weight, "data": data}
            successful_models.append(model_name)
            logger.debug(f"  ✓ {MODELS_CONFIG[model_name]['name']} (weight={weight:.0%})")
        except Exception as e:
            logger.warning(f"  ✗ {MODELS_CONFIG[model_name]['name']}: {e}")

        # Be polite to the API
        time.sleep(0.15)

    if not ensemble_data:
        raise ValueError(f"All models failed for ({lat:.4f}, {lon:.4f})")

    if len(ensemble_data) < len(weights):
        logger.info(
            f"  {len(ensemble_data)}/{len(weights)} models succeeded — re-normalising weights"
        )

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
            # Convert to float array — None elements become NaN automatically
            values = np.array(values, dtype=object)
            values = np.where(values == None, np.nan, values).astype(float)  # noqa: E711

            # Handle length mismatch: some high-res models have shorter
            # forecast horizons than global models. Pad with NaN.
            if len(values) < n_times:
                padded = np.full(n_times, np.nan)
                padded[:len(values)] = values
                values = padded
            elif len(values) > n_times:
                values = values[:n_times]

            if len(values) == n_times:
                # Only add weighted contribution where we have valid data
                valid_mask = ~np.isnan(values)
                ensemble_vars[var] += np.where(valid_mask, weight * values, 0.0)
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
    # Replace any remaining NaN values with 0.0 to prevent PostgreSQL errors
    df = df.fillna(0.0)
    df.attrs["models_used"] = model_names
    return df
