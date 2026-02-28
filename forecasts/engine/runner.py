"""
OrcaMet Forecast Engine — Runner

Orchestrates forecast generation for a single site:
1. Reads site location and thresholds from the database
2. Fetches multi-model ensemble from Open-Meteo
3. Computes hourly risk scores
4. Stores ForecastRun + HourlyForecast records
"""

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
from django.conf import settings
from django.utils import timezone as dj_timezone

from forecasts.models import ForecastRun, HourlyForecast
from sites.models import Site, ThresholdProfile
from .core import (
    fetch_ensemble,
    calculate_hourly_risk,
    get_recommendation,
)

logger = logging.getLogger(__name__)

# Work window defaults (can be overridden in settings)
WORK_START = getattr(settings, "FORECAST_WORK_START_HOUR", 7)
WORK_END = getattr(settings, "FORECAST_WORK_END_HOUR", 18)
NUM_DAYS = getattr(settings, "FORECAST_NUM_DAYS", 3)


def run_forecast_for_site(site: Site) -> list:
    """
    Generate forecasts for a single site for the next NUM_DAYS days.

    Returns a list of ForecastRun objects (one per day).
    """
    if not site.latitude or not site.longitude:
        logger.error(f"Site {site.name} has no coordinates — skipping")
        return []

    if not site.is_active or site.job_complete:
        logger.info(f"Site {site.name} is inactive or job complete — skipping")
        return []

    # Get active thresholds for this site
    threshold_profile = ThresholdProfile.objects.filter(
        site=site, is_active=True
    ).first()

    if threshold_profile:
        thresholds = threshold_profile.as_dict()
    else:
        # Use defaults if no threshold profile exists
        logger.warning(f"No active thresholds for {site.name} — using defaults")
        thresholds = {
            "wind_mean_caution": 10.0, "wind_mean_cancel": 14.0,
            "gust_caution": 15.0, "gust_cancel": 20.0,
            "precip_caution": 0.7, "precip_cancel": 2.0,
            "temp_min_caution": 1.0, "temp_min_cancel": -2.0,
        }

    today = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=NUM_DAYS - 1)

    logger.info(
        f"Generating forecast for {site.name} "
        f"({site.latitude:.4f}, {site.longitude:.4f}) "
        f"from {today} to {end_date}"
    )

    # Fetch the ensemble data (all days in one API call per model)
    try:
        hourly_df = fetch_ensemble(
            lat=site.latitude,
            lon=site.longitude,
            exposure=site.exposure,
            start_date=today.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        logger.error(f"Ensemble fetch failed for {site.name}: {e}")
        # Create a failed run record
        run = ForecastRun.objects.create(
            site=site,
            forecast_date=today,
            status=ForecastRun.Status.FAILED,
            error_message=str(e),
            thresholds_snapshot=thresholds,
        )
        return [run]

    models_used = hourly_df.attrs.get("models_used", [])
    logger.info(f"  ✓ Fetched {len(hourly_df)} hours from {len(models_used)} models")

    # Add hourly risk scores
    hourly_df["hourly_risk"] = hourly_df.apply(
        lambda row: calculate_hourly_risk(
            row["wind_speed"], row["wind_gusts"],
            row["precipitation"], row["temperature"],
            thresholds,
        ),
        axis=1,
    )

    # Split by date and create one ForecastRun per day
    hourly_df["date"] = hourly_df["time"].dt.date
    runs = []

    for forecast_date, day_data in hourly_df.groupby("date"):
        # Filter to work hours for summary stats
        work_hours = day_data[
            (day_data["time"].dt.hour >= WORK_START) &
            (day_data["time"].dt.hour <= WORK_END)
        ]

        if work_hours.empty:
            continue

        peak_risk = float(work_hours["hourly_risk"].max())
        peak_wind = float(work_hours["wind_speed"].max())
        peak_gust = float(work_hours["wind_gusts"].max())
        peak_precip = float(work_hours["precipitation"].max())
        min_temp = float(work_hours["temperature"].min())
        recommendation = get_recommendation(peak_risk)

        # Delete any existing runs for this site+date (replace with fresh)
        ForecastRun.objects.filter(
            site=site,
            forecast_date=forecast_date,
        ).delete()

        run = ForecastRun.objects.create(
            site=site,
            forecast_date=forecast_date,
            status=ForecastRun.Status.SUCCESS,
            peak_risk=peak_risk,
            recommendation=recommendation,
            peak_wind=peak_wind,
            peak_gust=peak_gust,
            peak_precip=peak_precip,
            min_temp=min_temp,
            models_used=[m for m in models_used],
            thresholds_snapshot=thresholds,
        )

        # Store all hourly data (full 24h, not just work hours)
        hourly_records = []
        for _, row in day_data.iterrows():
            hourly_records.append(HourlyForecast(
                run=run,
                timestamp=row["time"],
                wind_speed=float(row["wind_speed"]),
                wind_gusts=float(row["wind_gusts"]),
                precipitation=float(row["precipitation"]),
                temperature=float(row["temperature"]),
                wind_spread=float(row.get("wind_speed_spread", 0)),
                gust_spread=float(row.get("wind_gusts_spread", 0)),
                precip_spread=float(row.get("precipitation_spread", 0)),
                temp_spread=float(row.get("temperature_spread", 0)),
                hourly_risk=float(row["hourly_risk"]),
            ))

        HourlyForecast.objects.bulk_create(hourly_records)

        logger.info(
            f"  ✓ {forecast_date}: peak risk {peak_risk:.1f}% "
            f"[{recommendation}] — {len(hourly_records)} hours stored"
        )
        runs.append(run)

    return runs


def run_forecasts_all_active():
    """
    Run forecasts for ALL active sites that are not job_complete.
    Used by the cron/management command.
    """
    active_sites = Site.objects.filter(is_active=True, job_complete=False)
    total = active_sites.count()
    logger.info(f"Running forecasts for {total} active sites")

    all_runs = []
    for idx, site in enumerate(active_sites, 1):
        logger.info(f"[{idx}/{total}] {site.name}")
        try:
            runs = run_forecast_for_site(site)
            all_runs.extend(runs)
        except Exception as e:
            logger.error(f"  ✗ Failed for {site.name}: {e}", exc_info=True)

    success = sum(1 for r in all_runs if r.status == ForecastRun.Status.SUCCESS)
    failed = sum(1 for r in all_runs if r.status == ForecastRun.Status.FAILED)
    logger.info(f"Complete: {success} successful, {failed} failed forecast runs")

    return all_runs
