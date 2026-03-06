"""
OrcaMet Portal — risk_grid management command (Multi-Model Ensemble)

Fetches weather data for a UK grid from ALL eligible models using BATCH
API calls to Open-Meteo, blends with geographic-aware weighting (matching
the site-specific forecast engine in core.py), and stores hourly ensemble
risk scores for the interactive contour map.

Models: UKV, ICON-D2, AROME France, HARMONIE Europe, ARPEGE Europe,
        ECMWF IFS, ICON-EU

The number of API calls is approximately:
    (grid_points / batch_size) × eligible_models ≈ 8 × 7 = 56 calls
    (not 380 × 7 = 2,660 individual calls)

Usage:
    python manage.py risk_grid                    # Default 0.5° grid, all models
    python manage.py risk_grid --resolution 0.25  # Finer grid
    python manage.py risk_grid --days 2           # 2-day forecast
    python manage.py risk_grid --batch-size 30    # Smaller batches

After completion, run generate_contour_cache to pre-render map images:
    python manage.py generate_contour_cache
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
from django.core.management.base import BaseCommand
from django.conf import settings

from forecasts.models import UKRiskGridRun, UKRiskGridPoint
from forecasts.engine.core import (
    calculate_hourly_risk,
    get_model_weights,
    is_in_domain,
    MODELS_CONFIG,
)

logger = logging.getLogger(__name__)

# UK bounding box (covers mainland GB + Northern Ireland)
UK_LAT_MIN = 49.9
UK_LAT_MAX = 58.7
UK_LON_MIN = -7.6
UK_LON_MAX = 1.8

# Default thresholds for the grid (generic — no site-specific exposure)
DEFAULT_THRESHOLDS = {
    "wind_mean_caution": 10.0,
    "wind_mean_cancel": 14.0,
    "gust_caution": 15.0,
    "gust_cancel": 20.0,
    "precip_caution": 0.7,
    "precip_cancel": 2.0,
    "temp_min_caution": 1.0,
    "temp_min_cancel": -2.0,
}

# Weather variables to extract from the API
HOURLY_VARS = "wind_speed_10m,wind_gusts_10m,precipitation,temperature_2m"


def _parse_timestamp(t_str):
    """Parse Open-Meteo timestamp string to timezone-aware datetime."""
    if "T" in t_str:
        cleaned = t_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    else:
        return datetime.strptime(t_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _safe_float(value, default=0.0):
    """Convert a value to float, returning default for None/NaN/Inf."""
    if value is None:
        return default
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def fetch_batch(model_name, lats, lons, start_date, end_date):
    """
    Fetch weather data for MULTIPLE locations in a single API call.

    Open-Meteo accepts comma-separated latitude/longitude values and
    returns an array of results (one per location).

    Returns:
        List of dicts, one per location. Failed locations return None.
    """
    config = MODELS_CONFIG[model_name]
    api_key = getattr(settings, "OPENMETEO_API_KEY", "")

    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat in lats),
        "longitude": ",".join(f"{lon:.4f}" for lon in lons),
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "start_date": start_date,
        "end_date": end_date,
        **config["params"],
    }
    if api_key:
        params["apikey"] = api_key

    resp = requests.get(config["url"], params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Single location returns a dict; multiple returns a list
    if isinstance(data, dict):
        data = [data]

    results = []
    for i, item in enumerate(data):
        h = item.get("hourly", {})
        if not h or "time" not in h:
            results.append(None)
            continue
        results.append({
            "lat": lats[i],
            "lon": lons[i],
            "time": h["time"],
            "wind_speed": h.get("wind_speed_10m", []),
            "wind_gusts": h.get("wind_gusts_10m", []),
            "precipitation": h.get("precipitation", []),
            "temperature": h.get("temperature_2m", []),
        })

    return results


class Command(BaseCommand):
    help = "Generate UK-wide multi-model ensemble risk grid for the interactive contour map"

    def add_arguments(self, parser):
        parser.add_argument(
            "--resolution",
            type=float,
            default=0.5,
            help="Grid spacing in degrees (default: 0.5 ≈ 55km)",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=3,
            help="Number of forecast days (default: 3)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50,
            help="Locations per API call (default: 50, max ~100)",
        )

    def handle(self, *args, **options):
        resolution = options["resolution"]
        num_days = options["days"]
        batch_size = options["batch_size"]

        # Build the grid
        lats = np.arange(UK_LAT_MIN, UK_LAT_MAX + resolution, resolution)
        lons = np.arange(UK_LON_MIN, UK_LON_MAX + resolution, resolution)
        grid_points = [
            (round(float(lat), 4), round(float(lon), 4))
            for lat in lats
            for lon in lons
        ]
        total_points = len(grid_points)

        today = datetime.now(timezone.utc).date()
        end_date = today + timedelta(days=num_days - 1)
        start_str = today.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Determine which models cover which grid points
        model_points = {}  # model_name -> list of (lat, lon)
        for model_name in MODELS_CONFIG:
            in_domain = [
                (lat, lon) for lat, lon in grid_points
                if is_in_domain(model_name, lat, lon)
            ]
            if in_domain:
                model_points[model_name] = in_domain

        # Calculate total API calls
        total_api_calls = sum(
            (len(pts) + batch_size - 1) // batch_size
            for pts in model_points.values()
        )

        self.stdout.write(
            f"Generating UK ensemble risk grid: {len(lats)}×{len(lons)} = "
            f"{total_points} points at {resolution}° resolution"
        )
        self.stdout.write(f"  Period: {today} to {end_date} ({num_days} days)")
        self.stdout.write(f"  Models: {len(model_points)} eligible")
        for m, pts in model_points.items():
            n_batches = (len(pts) + batch_size - 1) // batch_size
            self.stdout.write(
                f"    {MODELS_CONFIG[m]['name']}: "
                f"{len(pts)} pts → {n_batches} API calls"
            )
        self.stdout.write(
            f"  Total API calls: {total_api_calls} "
            f"(vs {total_points * len(model_points)} individual)"
        )

        # Create the run record
        models_used = list(model_points.keys())
        grid_run = UKRiskGridRun.objects.create(
            forecast_date=today,
            status=UKRiskGridRun.Status.RUNNING,
            lat_min=UK_LAT_MIN,
            lat_max=UK_LAT_MAX,
            lon_min=UK_LON_MIN,
            lon_max=UK_LON_MAX,
            resolution=resolution,
            grid_points=total_points,
            models_used=models_used,
        )

        # Delete previous successful runs for today (replace strategy)
        UKRiskGridRun.objects.filter(
            forecast_date=today,
            status=UKRiskGridRun.Status.SUCCESS,
        ).exclude(pk=grid_run.pk).delete()

        start_time = time.time()

        # ============================================================
        # PHASE 1: Fetch raw data from all models
        # ============================================================
        # model_data[model_name][(lat, lon)] = {time: [...], wind_speed: [...], ...}
        model_data = {}
        failed_models = []
        api_calls_made = 0

        for model_name, points in model_points.items():
            model_display = MODELS_CONFIG[model_name]["name"]
            n_batches = (len(points) + batch_size - 1) // batch_size

            self.stdout.write(
                f"\n  Fetching {model_display} "
                f"({len(points)} pts, {n_batches} batches)..."
            )

            model_results = {}
            model_failed = 0

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(points))
                batch_pts = points[batch_start:batch_end]
                batch_lats = [p[0] for p in batch_pts]
                batch_lons = [p[1] for p in batch_pts]

                try:
                    results = fetch_batch(
                        model_name, batch_lats, batch_lons,
                        start_str, end_str
                    )
                    api_calls_made += 1

                    for result in results:
                        if result is None:
                            model_failed += 1
                            continue
                        key = (result["lat"], result["lon"])
                        model_results[key] = result

                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        # Rate limited — wait and retry once
                        self.stdout.write(
                            self.style.WARNING(
                                f"    ⚠ Rate limited on {model_display}! "
                                f"Waiting 30s..."
                            )
                        )
                        time.sleep(30)
                        try:
                            results = fetch_batch(
                                model_name, batch_lats, batch_lons,
                                start_str, end_str
                            )
                            api_calls_made += 1
                            for result in results:
                                if result is None:
                                    model_failed += 1
                                    continue
                                key = (result["lat"], result["lon"])
                                model_results[key] = result
                        except Exception as retry_e:
                            self.stdout.write(
                                self.style.ERROR(
                                    f"    ✗ Retry failed: {retry_e}"
                                )
                            )
                            model_failed += len(batch_pts)
                    else:
                        self.stdout.write(
                            self.style.ERROR(
                                f"    ✗ Batch {batch_idx + 1} failed: {e}"
                            )
                        )
                        model_failed += len(batch_pts)

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"    ✗ Batch {batch_idx + 1} failed: {e}"
                        )
                    )
                    model_failed += len(batch_pts)

                # Pause between batches
                time.sleep(0.5)

            if model_results:
                model_data[model_name] = model_results
                self.stdout.write(
                    self.style.SUCCESS(
                        f"    ✓ {model_display}: {len(model_results)} pts OK"
                        + (f", {model_failed} failed" if model_failed else "")
                    )
                )
            else:
                failed_models.append(model_name)
                self.stdout.write(
                    self.style.ERROR(
                        f"    ✗ {model_display}: ALL points failed"
                    )
                )

        if not model_data:
            grid_run.status = UKRiskGridRun.Status.FAILED
            grid_run.error_message = "All models failed — no data fetched"
            grid_run.save()
            self.stderr.write(self.style.ERROR("  ✗ All models failed"))
            return

        # ============================================================
        # PHASE 2: Blend models with geographic-aware weighting
        # ============================================================
        self.stdout.write(
            f"\n  Blending {len(model_data)} models with geographic weighting..."
        )

        # Determine the reference timestamp list from the model with
        # the most timestamps (usually ECMWF with the longest horizon)
        ref_model = max(
            model_data.keys(),
            key=lambda m: max(
                len(d.get("time", []))
                for d in model_data[m].values()
            )
        )
        # Get timestamps from first available point of reference model
        ref_point = next(iter(model_data[ref_model].values()))
        ref_times = ref_point["time"]

        all_point_records = []
        blend_errors = 0

        for lat, lon in grid_points:
            key = (lat, lon)

            # Get geographic weights for this location
            weights = get_model_weights(lat, lon, exposure="urban")

            # Remove models that failed entirely
            weights = {
                m: w for m, w in weights.items()
                if m in model_data
            }

            if not weights:
                # Fallback: use whatever models we have
                available = [m for m in model_data if key in model_data[m]]
                if not available:
                    blend_errors += 1
                    continue
                weights = {m: 1.0 / len(available) for m in available}

            # Re-normalise weights for models that have data at this point
            active_weights = {
                m: w for m, w in weights.items()
                if m in model_data and key in model_data[m]
            }

            if not active_weights:
                blend_errors += 1
                continue

            total_w = sum(active_weights.values())
            active_weights = {m: w / total_w for m, w in active_weights.items()}

            # Blend across timestamps
            for t_idx, t_str in enumerate(ref_times):
                w_blend = 0.0
                g_blend = 0.0
                p_blend = 0.0
                t_blend = 0.0
                w_total = 0.0

                for m_name, m_weight in active_weights.items():
                    m_point = model_data[m_name].get(key)
                    if m_point is None:
                        continue

                    m_times = m_point.get("time", [])
                    if t_idx >= len(m_times):
                        # This model's forecast horizon is shorter
                        continue

                    ws = m_point.get("wind_speed", [])
                    gs = m_point.get("wind_gusts", [])
                    ps = m_point.get("precipitation", [])
                    ts = m_point.get("temperature", [])

                    w_val = _safe_float(
                        ws[t_idx] if t_idx < len(ws) else None
                    )
                    g_val = _safe_float(
                        gs[t_idx] if t_idx < len(gs) else None
                    )
                    p_val = _safe_float(
                        ps[t_idx] if t_idx < len(ps) else None
                    )
                    t_val = _safe_float(
                        ts[t_idx] if t_idx < len(ts) else None, default=10.0
                    )

                    w_blend += m_weight * w_val
                    g_blend += m_weight * g_val
                    p_blend += m_weight * p_val
                    t_blend += m_weight * t_val
                    w_total += m_weight

                # Re-normalise if some models had shorter horizons
                if w_total > 0 and w_total < 0.99:
                    w_blend /= w_total
                    g_blend /= w_total
                    p_blend /= w_total
                    t_blend /= w_total
                elif w_total == 0:
                    continue

                risk = calculate_hourly_risk(
                    w_blend, g_blend, p_blend, t_blend, DEFAULT_THRESHOLDS
                )

                all_point_records.append(UKRiskGridPoint(
                    run=grid_run,
                    latitude=lat,
                    longitude=lon,
                    timestamp=_parse_timestamp(t_str),
                    wind_speed=round(w_blend, 2),
                    wind_gusts=round(g_blend, 2),
                    precipitation=round(p_blend, 2),
                    temperature=round(t_blend, 2),
                    risk=round(risk, 2),
                ))

        # ============================================================
        # PHASE 3: Store results
        # ============================================================
        if all_point_records:
            self.stdout.write(
                f"  Storing {len(all_point_records)} ensemble grid records..."
            )
            try:
                batch_db_size = 5000
                for i in range(0, len(all_point_records), batch_db_size):
                    batch = all_point_records[i:i + batch_db_size]
                    UKRiskGridPoint.objects.bulk_create(batch)

                grid_run.status = UKRiskGridRun.Status.SUCCESS
                successful = total_points - blend_errors
                grid_run.num_hours = (
                    len(all_point_records) // max(successful, 1)
                )
                grid_run.save()

                elapsed = time.time() - start_time
                self.stdout.write(self.style.SUCCESS(
                    f"\n  ✓ Complete: {len(all_point_records)} records "
                    f"({successful} points, {blend_errors} failed) "
                    f"in {elapsed:.0f}s using {api_calls_made} API calls "
                    f"across {len(model_data)} models"
                ))
                self.stdout.write(
                    f"  Models used: "
                    f"{', '.join(MODELS_CONFIG[m]['name'] for m in model_data)}"
                )

            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                grid_run.status = UKRiskGridRun.Status.FAILED
                grid_run.error_message = str(e)
                grid_run.save()
                self.stderr.write(
                    self.style.ERROR(f"  ✗ Storage failed: {e}")
                )
        else:
            grid_run.status = UKRiskGridRun.Status.FAILED
            grid_run.error_message = "No data produced after blending"
            grid_run.save()
            self.stderr.write(
                self.style.ERROR("  ✗ No data produced after blending")
            )
