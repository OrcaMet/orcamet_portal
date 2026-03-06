"""
OrcaMet Portal — Dashboard Views

Main views for logged-in users: dashboard overview and site detail with
live forecast data, charts, and risk assessments.
"""

import io
import json
from collections import defaultdict
from datetime import date, timedelta

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404
from django.utils import timezone

from forecasts.models import ForecastRun, HourlyForecast
from sites.models import Site


def _get_user_sites(user):
    """Return the queryset of sites visible to a user."""
    if user.is_superadmin:
        return Site.objects.filter(is_active=True).select_related("client")
    elif user.client:
        return Site.objects.filter(
            client=user.client, is_active=True
        ).select_related("client")
    return Site.objects.none()


def _latest_run_for_site(site):
    """Return the most recent successful ForecastRun for a site."""
    return (
        ForecastRun.objects.filter(site=site, status=ForecastRun.Status.SUCCESS)
        .order_by("-forecast_date", "-generated_at")
        .first()
    )


def _annotate_sites_with_forecasts(sites_qs):
    """Attach latest forecast info to each site object for template use."""
    annotated = []
    today = date.today()
    for site in sites_qs:
        run = (
            ForecastRun.objects.filter(
                site=site,
                status=ForecastRun.Status.SUCCESS,
                forecast_date=today,
            )
            .order_by("-generated_at")
            .first()
        )
        if run is None:
            run = _latest_run_for_site(site)

        site.latest_run = run
        annotated.append(site)
    return annotated


def _build_chart_data(site, forecast_days):
    """
    Build the hourly chart data as a Python dict, ready to be serialised
    into the template as inline JSON.
    """
    from sites.models import ThresholdProfile

    if not forecast_days:
        return json.dumps({"hourly": [], "thresholds": {}})

    run_ids = [run.pk for run in forecast_days]
    hourly_qs = (
        HourlyForecast.objects.filter(run_id__in=run_ids)
        .order_by("timestamp")
        .values(
            "timestamp",
            "wind_speed",
            "wind_gusts",
            "precipitation",
            "temperature",
            "wind_spread",
            "gust_spread",
            "precip_spread",
            "temp_spread",
            "hourly_risk",
        )
    )

    threshold = ThresholdProfile.objects.filter(site=site, is_active=True).first()
    thresholds = (
        threshold.as_dict()
        if threshold
        else {
            "wind_mean_caution": 10.0,
            "wind_mean_cancel": 14.0,
            "gust_caution": 15.0,
            "gust_cancel": 20.0,
            "precip_caution": 0.7,
            "precip_cancel": 2.0,
            "temp_min_caution": 1.0,
            "temp_min_cancel": -2.0,
        }
    )

    hourly_list = [
        {
            "time": h["timestamp"].isoformat(),
            "wind_speed": round(h["wind_speed"], 1),
            "wind_gusts": round(h["wind_gusts"], 1),
            "precipitation": round(h["precipitation"], 1),
            "temperature": round(h["temperature"], 1),
            "wind_spread": round(h["wind_spread"], 1),
            "gust_spread": round(h["gust_spread"], 1),
            "precip_spread": round(h["precip_spread"], 1),
            "temp_spread": round(h["temp_spread"], 1),
            "risk": round(h["hourly_risk"], 1),
        }
        for h in hourly_qs
    ]

    data = {
        "site": {
            "name": site.name,
            "postcode": site.postcode,
            "exposure": site.get_exposure_display(),
        },
        "thresholds": thresholds,
        "hourly": hourly_list,
        "debug": {
            "run_ids": run_ids,
            "hourly_count": len(hourly_list),
        },
    }
    return json.dumps(data)


@login_required(login_url="/login/")
def home(request):
    """Main dashboard view with live forecast data."""
    user = request.user
    sites_qs = _get_user_sites(user)
    sites_list = _annotate_sites_with_forecasts(sites_qs)

    total_sites = len(sites_list)
    sites_with_forecasts = sum(1 for s in sites_list if s.latest_run)
    alerts = sum(
        1
        for s in sites_list
        if s.latest_run
        and s.latest_run.recommendation in ("CAUTION", "CANCEL")
    )

    latest_ts = None
    for s in sites_list:
        if s.latest_run:
            if latest_ts is None or s.latest_run.generated_at > latest_ts:
                latest_ts = s.latest_run.generated_at

    context = {
        "user": user,
        "sites": sites_list,
        "site_count": total_sites,
        "forecast_count": sites_with_forecasts,
        "alert_count": alerts,
        "latest_forecast_time": latest_ts,
    }
    return render(request, "dashboard/home.html", context)


@login_required(login_url="/login/")
def site_detail(request, site_id):
    """
    Site detail view with full forecast display.
    Hourly data is embedded as inline JSON — no separate AJAX call needed.
    """
    user = request.user
    if user.is_superadmin:
        site = get_object_or_404(Site, pk=site_id, is_active=True)
    elif user.client:
        site = get_object_or_404(
            Site, pk=site_id, client=user.client, is_active=True
        )
    else:
        return render(request, "dashboard/no_access.html", status=403)

    today = date.today()
    runs = ForecastRun.objects.filter(
        site=site,
        status=ForecastRun.Status.SUCCESS,
        forecast_date__gte=today,
    ).order_by("forecast_date", "-generated_at")

    seen_dates = set()
    forecast_days = []
    for run in runs:
        if run.forecast_date not in seen_dates:
            seen_dates.add(run.forecast_date)
            forecast_days.append(run)

    from sites.models import ThresholdProfile

    threshold = ThresholdProfile.objects.filter(site=site, is_active=True).first()
    chart_data_json = _build_chart_data(site, forecast_days)

    context = {
        "user": user,
        "site": site,
        "forecast_days": forecast_days,
        "threshold": threshold,
        "today": today,
        "chart_data_json": chart_data_json,
    }
    return render(request, "dashboard/site_detail.html", context)


@login_required(login_url="/login/")
def weather_map(request):
    """
    Interactive Leaflet map showing all sites with live risk status
    and ensemble contour overlay.
    """
    user = request.user
    sites_qs = _get_user_sites(user)
    sites_list = _annotate_sites_with_forecasts(sites_qs)

    total_sites = len(sites_list)
    go_count = sum(
        1
        for s in sites_list
        if s.latest_run and s.latest_run.recommendation == "GO"
    )
    caution_count = sum(
        1
        for s in sites_list
        if s.latest_run and s.latest_run.recommendation == "CAUTION"
    )
    cancel_count = sum(
        1
        for s in sites_list
        if s.latest_run and s.latest_run.recommendation == "CANCEL"
    )
    pending_count = sum(1 for s in sites_list if not s.latest_run)

    context = {
        "user": user,
        "total_sites": total_sites,
        "go_count": go_count,
        "caution_count": caution_count,
        "cancel_count": cancel_count,
        "pending_count": pending_count,
    }
    return render(request, "dashboard/weather_map.html", context)


@login_required(login_url="/login/")
def map_sites_json(request):
    """
    JSON API endpoint returning all visible sites with coordinates
    and latest forecast data for the Leaflet map.
    """
    user = request.user
    sites_qs = _get_user_sites(user)
    sites_list = _annotate_sites_with_forecasts(sites_qs)

    features = []
    for site in sites_list:
        if not site.latitude or not site.longitude:
            continue

        run = site.latest_run
        props = {
            "id": site.pk,
            "name": site.name,
            "client": site.client.name,
            "postcode": site.postcode,
            "exposure": site.get_exposure_display(),
            "job_complete": site.job_complete,
            "has_forecast": run is not None,
        }

        if run:
            props.update(
                {
                    "recommendation": run.recommendation,
                    "peak_risk": (
                        round(run.peak_risk, 1)
                        if run.peak_risk is not None
                        else None
                    ),
                    "peak_wind": (
                        round(run.peak_wind, 1)
                        if run.peak_wind is not None
                        else None
                    ),
                    "peak_gust": (
                        round(run.peak_gust, 1)
                        if run.peak_gust is not None
                        else None
                    ),
                    "peak_precip": (
                        round(run.peak_precip, 1)
                        if run.peak_precip is not None
                        else None
                    ),
                    "min_temp": (
                        round(run.min_temp, 1)
                        if run.min_temp is not None
                        else None
                    ),
                    "forecast_date": run.forecast_date.isoformat(),
                    "generated_at": run.generated_at.isoformat(),
                }
            )
        else:
            props.update(
                {
                    "recommendation": "PENDING",
                    "peak_risk": None,
                }
            )

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [site.longitude, site.latitude],
                },
                "properties": props,
            }
        )

    return JsonResponse(
        {
            "type": "FeatureCollection",
            "features": features,
        }
    )


@login_required(login_url="/login/")
def forecast_chart_data(request, site_id):
    """JSON API endpoint — debug tool for inspecting raw forecast data."""
    user = request.user
    if user.is_superadmin:
        site = get_object_or_404(Site, pk=site_id, is_active=True)
    elif user.client:
        site = get_object_or_404(
            Site, pk=site_id, client=user.client, is_active=True
        )
    else:
        return JsonResponse({"error": "Access denied"}, status=403)

    today = date.today()
    runs = ForecastRun.objects.filter(
        site=site,
        status=ForecastRun.Status.SUCCESS,
        forecast_date__gte=today - timedelta(days=1),
    ).order_by("forecast_date", "-generated_at")

    seen = set()
    run_ids = []
    for run in runs:
        if run.forecast_date not in seen:
            seen.add(run.forecast_date)
            run_ids.append(run.pk)

    hourly = (
        HourlyForecast.objects.filter(run_id__in=run_ids)
        .order_by("timestamp")
        .values(
            "timestamp",
            "wind_speed",
            "wind_gusts",
            "precipitation",
            "temperature",
            "wind_spread",
            "gust_spread",
            "precip_spread",
            "temp_spread",
            "hourly_risk",
        )
    )

    from sites.models import ThresholdProfile

    threshold = ThresholdProfile.objects.filter(site=site, is_active=True).first()
    thresholds = (
        threshold.as_dict()
        if threshold
        else {
            "wind_mean_caution": 10.0,
            "wind_mean_cancel": 14.0,
            "gust_caution": 15.0,
            "gust_cancel": 20.0,
            "precip_caution": 0.7,
            "precip_cancel": 2.0,
            "temp_min_caution": 1.0,
            "temp_min_cancel": -2.0,
        }
    )

    hourly_list = list(hourly)
    data = {
        "site": {
            "name": site.name,
            "postcode": site.postcode,
            "exposure": site.get_exposure_display(),
        },
        "thresholds": thresholds,
        "debug": {
            "run_ids": run_ids,
            "hourly_count": len(hourly_list),
        },
        "hourly": [
            {
                "time": h["timestamp"].isoformat(),
                "wind_speed": round(h["wind_speed"], 1),
                "wind_gusts": round(h["wind_gusts"], 1),
                "precipitation": round(h["precipitation"], 1),
                "temperature": round(h["temperature"], 1),
                "wind_spread": round(h["wind_spread"], 1),
                "gust_spread": round(h["gust_spread"], 1),
                "precip_spread": round(h["precip_spread"], 1),
                "temp_spread": round(h["temp_spread"], 1),
                "risk": round(h["hourly_risk"], 1),
            }
            for h in hourly_list
        ],
    }
    return JsonResponse(data)


@login_required(login_url="/login/")
def map_sites_hourly_json(request):
    """
    JSON API endpoint returning all visible sites with their full hourly
    forecast timeseries. Used by the map time slider.
    """
    user = request.user
    sites_qs = _get_user_sites(user)
    today = date.today()

    sites_with_coords = sites_qs.filter(
        latitude__isnull=False, longitude__isnull=False
    ).select_related("client")

    site_run_map = {}
    for site in sites_with_coords:
        run = (
            ForecastRun.objects.filter(
                site=site,
                status=ForecastRun.Status.SUCCESS,
                forecast_date__gte=today,
            )
            .order_by("forecast_date", "-generated_at")
            .first()
        )
        if run:
            site_run_map[site.pk] = (site, run)

    if not site_run_map:
        return JsonResponse({"timestamps": [], "frames": {}})

    run_ids = [run.pk for _, run in site_run_map.values()]
    hourly_qs = (
        HourlyForecast.objects.filter(run_id__in=run_ids)
        .order_by("timestamp")
        .values(
            "run__site_id",
            "timestamp",
            "wind_speed",
            "wind_gusts",
            "precipitation",
            "temperature",
            "hourly_risk",
        )
    )

    frames = defaultdict(list)
    for h in hourly_qs:
        site_id = h["run__site_id"]
        if site_id not in site_run_map:
            continue

        site, run = site_run_map[site_id]
        ts = h["timestamp"].isoformat()
        risk = round(h["hourly_risk"], 1)

        if risk < 20:
            rec = "GO"
        elif risk < 50:
            rec = "CAUTION"
        else:
            rec = "CANCEL"

        frames[ts].append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [site.longitude, site.latitude],
                },
                "properties": {
                    "id": site.pk,
                    "name": site.name,
                    "client": site.client.name,
                    "postcode": site.postcode,
                    "recommendation": rec,
                    "risk": risk,
                    "wind_speed": round(h["wind_speed"], 1),
                    "wind_gusts": round(h["wind_gusts"], 1),
                    "precipitation": round(h["precipitation"], 1),
                    "temperature": round(h["temperature"], 1),
                },
            }
        )

    timestamps = sorted(frames.keys())
    return JsonResponse(
        {
            "timestamps": timestamps,
            "frames": {
                ts: {"type": "FeatureCollection", "features": frames[ts]}
                for ts in timestamps
            },
        }
    )


@login_required(login_url="/login/")
def map_risk_grid_json(request):
    """
    JSON API endpoint returning the UK-wide risk grid data.
    Returns {available: false} gracefully if no data exists.
    """
    try:
        from forecasts.models import UKRiskGridRun, UKRiskGridPoint
    except Exception:
        return JsonResponse(
            {"available": False, "message": "Risk grid models not available"}
        )

    try:
        grid_run = (
            UKRiskGridRun.objects.filter(status=UKRiskGridRun.Status.SUCCESS)
            .order_by("-forecast_date", "-generated_at")
            .first()
        )
    except Exception:
        return JsonResponse(
            {"available": False, "message": "Risk grid not yet configured"}
        )

    if not grid_run:
        return JsonResponse(
            {"available": False, "message": "No risk grid data available"}
        )

    requested_ts = request.GET.get("timestamp")
    if requested_ts:
        from django.utils.dateparse import parse_datetime

        ts = parse_datetime(requested_ts)
        if not ts:
            return JsonResponse({"error": "Invalid timestamp"}, status=400)

        points = UKRiskGridPoint.objects.filter(
            run=grid_run, timestamp=ts
        ).values(
            "latitude",
            "longitude",
            "risk",
            "wind_speed",
            "wind_gusts",
            "precipitation",
            "temperature",
        )

        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [p["longitude"], p["latitude"]],
                },
                "properties": {
                    "risk": round(p["risk"], 1),
                    "wind": round(p["wind_speed"], 1),
                    "gust": round(p["wind_gusts"], 1),
                    "precip": round(p["precipitation"], 1),
                    "temp": round(p["temperature"], 1),
                },
            }
            for p in points
        ]
        return JsonResponse(
            {
                "type": "FeatureCollection",
                "features": features,
                "timestamp": requested_ts,
            }
        )
    else:
        timestamps = list(
            UKRiskGridPoint.objects.filter(run=grid_run)
            .values_list("timestamp", flat=True)
            .distinct()
            .order_by("timestamp")
        )
        return JsonResponse(
            {
                "available": True,
                "forecast_date": grid_run.forecast_date.isoformat(),
                "generated_at": grid_run.generated_at.isoformat(),
                "resolution": grid_run.resolution,
                "grid_points": grid_run.grid_points,
                "timestamps": [t.isoformat() for t in timestamps],
            }
        )


# ================================================================
# Contour map views (server-rendered CloughTocher2D interpolation)
# ================================================================

import logging

logger = logging.getLogger(__name__)


@login_required(login_url="/login/")
def map_contour_image(request):
    """
    Serve a contour map as a transparent PNG image.

    For TIMESTAMP requests:
      - Serves pre-cached image from CachedContourImage (instant)
      - Returns 404 if no cache exists (run generate_contour_cache)
      - Does NOT fall back to on-the-fly rendering (prevents hangs)

    For PEAK requests (no timestamp):
      - Renders on-the-fly (one-off, acceptable latency)

    Query params:
        timestamp (str): ISO timestamp. Required for hourly frames.
        var (str):       'risk', 'wind', 'gust', 'precip', 'temp'. Default: 'risk'.
        resolution (int): Interpolation resolution. Default: 300.
    """
    from forecasts.models import (
        UKRiskGridRun,
        UKRiskGridPoint,
        CachedContourImage,
    )
    from django.db.models import Max, Min
    import numpy as np

    var_name = request.GET.get("var", "risk")
    requested_ts = request.GET.get("timestamp")
    resolution = int(request.GET.get("resolution", "300"))

    # Find latest successful grid run
    grid_run = (
        UKRiskGridRun.objects.filter(status=UKRiskGridRun.Status.SUCCESS)
        .order_by("-generated_at")
        .first()
    )
    if not grid_run:
        return HttpResponse(
            "No grid data available", status=404, content_type="text/plain"
        )

    # ---- TIMESTAMP REQUEST: serve from cache only ----
    if requested_ts:
        from django.utils.dateparse import parse_datetime

        ts = parse_datetime(requested_ts)
        if not ts:
            return HttpResponse(
                "Invalid timestamp", status=400, content_type="text/plain"
            )

        # Look up cached image
        cached = CachedContourImage.objects.filter(
            run=grid_run,
            timestamp=ts,
            variable=var_name,
        ).first()

        if cached and cached.image_data:
            # Serve cached PNG — instant!
            response = HttpResponse(
                bytes(cached.image_data),
                content_type="image/png",
            )
            response["Cache-Control"] = "public, max-age=3600"
            return response

        # No cache — return 404 instead of attempting on-the-fly render
        # (which can hang due to coastline download or matplotlib overhead)
        logger.warning(
            f"No cached contour for {var_name} @ {requested_ts}. "
            f"Run: python manage.py generate_contour_cache"
        )
        return HttpResponse(
            "Contour not cached. Run generate_contour_cache.",
            status=404,
            content_type="text/plain",
        )

    # ---- PEAK REQUEST (no timestamp): on-the-fly is OK ----
    try:
        from forecasts.engine.map_interpolation import render_contour_to_bytes
    except ImportError as e:
        return HttpResponse(
            f"Map dependencies not installed: {e}",
            status=500,
            content_type="text/plain",
        )

    FIELD_MAP = {
        "wind": "wind_speed",
        "gust": "wind_gusts",
        "precip": "precipitation",
        "temp": "temperature",
        "risk": "risk",
    }
    field_name = FIELD_MAP.get(var_name, "risk")

    points_qs = UKRiskGridPoint.objects.filter(run=grid_run)

    if var_name == "temp":
        agg = points_qs.values("latitude", "longitude").annotate(
            val=Min(field_name)
        )
    else:
        agg = points_qs.values("latitude", "longitude").annotate(
            val=Max(field_name)
        )

    lats = np.array([p["latitude"] for p in agg])
    lons = np.array([p["longitude"] for p in agg])
    values = np.array([p["val"] for p in agg])

    if len(lats) < 4:
        return HttpResponse(
            "Not enough data points", status=404, content_type="text/plain"
        )

    try:
        png_bytes = render_contour_to_bytes(
            lats, lons, values,
            variable=var_name,
            resolution=resolution,
        )
    except Exception as e:
        logger.error(f"Contour rendering failed: {e}")
        return HttpResponse(
            f"Rendering failed: {e}", status=500, content_type="text/plain"
        )

    response = HttpResponse(png_bytes, content_type="image/png")
    response["Cache-Control"] = "public, max-age=300"
    return response


@login_required(login_url="/login/")
def map_contour_timestamps(request):
    """
    JSON endpoint returning available timestamps for the contour map
    time slider, plus cache availability status.
    """
    from forecasts.models import UKRiskGridRun, UKRiskGridPoint, CachedContourImage

    grid_run = (
        UKRiskGridRun.objects.filter(status=UKRiskGridRun.Status.SUCCESS)
        .order_by("-generated_at")
        .first()
    )

    if not grid_run:
        return JsonResponse({"available": False})

    timestamps = list(
        UKRiskGridPoint.objects.filter(run=grid_run)
        .values_list("timestamp", flat=True)
        .distinct()
        .order_by("timestamp")
    )

    # Check if contour cache exists for this run
    cache_count = CachedContourImage.objects.filter(run=grid_run).count()
    has_cache = cache_count > 0

    return JsonResponse(
        {
            "available": True,
            "forecast_date": grid_run.forecast_date.isoformat(),
            "generated_at": grid_run.generated_at.isoformat(),
            "grid_points": grid_run.grid_points,
            "models_used": grid_run.models_used,
            "has_cache": has_cache,
            "cache_count": cache_count,
            "timestamps": [
                t.isoformat() if hasattr(t, "isoformat") else str(t)
                for t in timestamps
            ],
        }
    )
