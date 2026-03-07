"""
OrcaMet Portal — generate_contour_cache management command

Pre-renders contour map PNGs for every (timestamp × variable) combination
from the latest risk grid run, storing them in the CachedContourImage model
for instant map animation.

MEMORY-SAFE: Flushes rendered images to the database after each variable
and calls gc.collect() after every render to prevent matplotlib memory leaks.

Usage:
    python manage.py generate_contour_cache                  # Latest run
    python manage.py generate_contour_cache --resolution 200 # Faster
    python manage.py generate_contour_cache --variables risk wind
"""

import gc
import logging
import time
from datetime import datetime, timezone

import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Max, Min

from forecasts.models import UKRiskGridRun, UKRiskGridPoint, CachedContourImage

logger = logging.getLogger(__name__)

# Variables to pre-render
ALL_VARIABLES = ["risk", "wind", "gust", "precip", "temp"]

# Map variable name -> model field
VARIABLE_FIELD_MAP = {
    "risk": "risk",
    "wind": "wind_speed",
    "gust": "wind_gusts",
    "precip": "precipitation",
    "temp": "temperature",
}


class Command(BaseCommand):
    help = "Pre-render contour map PNGs for instant map animation"

    def add_arguments(self, parser):
        parser.add_argument(
            "--resolution", type=int, default=200,
            help="Interpolation resolution (default: 200). Lower = faster + less memory.",
        )
        parser.add_argument(
            "--variables", nargs="+", default=ALL_VARIABLES,
            help=f"Variables to render (default: {' '.join(ALL_VARIABLES)})",
        )
        parser.add_argument(
            "--run-id", type=int, default=None,
            help="Specific UKRiskGridRun ID (default: latest successful)",
        )
        parser.add_argument(
            "--dpi", type=int, default=100,
            help="Image DPI (default: 100). Lower = less memory.",
        )

    def handle(self, *args, **options):
        resolution = options["resolution"]
        variables = options["variables"]
        run_id = options["run_id"]
        dpi = options["dpi"]

        # Lazy import to avoid loading matplotlib at startup
        from forecasts.engine.map_interpolation import render_contour_to_bytes

        # Find the grid run
        if run_id:
            try:
                grid_run = UKRiskGridRun.objects.get(pk=run_id)
            except UKRiskGridRun.DoesNotExist:
                self.stderr.write(self.style.ERROR(f"Grid run {run_id} not found"))
                return
        else:
            grid_run = (
                UKRiskGridRun.objects.filter(status=UKRiskGridRun.Status.SUCCESS)
                .order_by("-generated_at")
                .first()
            )

        if not grid_run:
            self.stderr.write(self.style.ERROR("No successful grid run found"))
            return

        # Get all unique timestamps
        timestamps = list(
            UKRiskGridPoint.objects.filter(run=grid_run)
            .values_list("timestamp", flat=True)
            .distinct()
            .order_by("timestamp")
        )

        if not timestamps:
            self.stderr.write(self.style.ERROR(f"No grid points for run {grid_run.pk}"))
            return

        total_images = len(timestamps) * len(variables)
        self.stdout.write(
            f"Pre-rendering contour cache for grid run {grid_run.pk}\n"
            f"  Forecast date: {grid_run.forecast_date}\n"
            f"  Timestamps: {len(timestamps)}\n"
            f"  Variables: {', '.join(variables)}\n"
            f"  Total images: {total_images}\n"
            f"  Resolution: {resolution}, DPI: {dpi}"
        )

        # Delete existing cache for this run
        deleted_count, _ = CachedContourImage.objects.filter(run=grid_run).delete()
        if deleted_count:
            self.stdout.write(f"  Cleared {deleted_count} existing cached images")

        start_time = time.time()
        total_rendered = 0
        total_failed = 0

        # Process ONE VARIABLE AT A TIME, flushing to DB after each
        for var_idx, var_name in enumerate(variables):
            field_name = VARIABLE_FIELD_MAP.get(var_name, "risk")

            self.stdout.write(
                f"\n  [{var_idx + 1}/{len(variables)}] Rendering {var_name}..."
            )

            var_records = []
            var_rendered = 0
            var_failed = 0

            for ts_idx, timestamp in enumerate(timestamps):
                try:
                    # Query grid points for this timestamp
                    points = UKRiskGridPoint.objects.filter(
                        run=grid_run, timestamp=timestamp,
                    )

                    lats = np.array(list(points.values_list("latitude", flat=True)))
                    lons = np.array(list(points.values_list("longitude", flat=True)))
                    values = np.array(list(points.values_list(field_name, flat=True)))

                    if len(lats) < 4:
                        var_failed += 1
                        continue

                    # Render the contour image
                    png_bytes = render_contour_to_bytes(
                        lats, lons, values,
                        variable=var_name,
                        resolution=resolution,
                        dpi=dpi,
                    )

                    var_records.append(CachedContourImage(
                        run=grid_run,
                        timestamp=timestamp,
                        variable=var_name,
                        image_data=png_bytes,
                    ))
                    var_rendered += 1

                    # Force garbage collection every render to combat
                    # matplotlib memory leaks
                    gc.collect()

                    # Progress every 6 hours
                    if (ts_idx + 1) % 6 == 0 or ts_idx == len(timestamps) - 1:
                        elapsed = time.time() - start_time
                        done = total_rendered + total_failed + var_rendered + var_failed
                        rate = done / elapsed if elapsed > 0 else 0
                        remaining = total_images - done
                        eta = remaining / rate if rate > 0 else 0
                        self.stdout.write(
                            f"    {ts_idx + 1}/{len(timestamps)} hours — "
                            f"{var_rendered} OK, {var_failed} failed, "
                            f"ETA {eta:.0f}s"
                        )

                except Exception as e:
                    logger.error(f"Failed to render {var_name} @ {timestamp}: {e}")
                    self.stdout.write(self.style.WARNING(
                        f"    ⚠ {var_name} @ hour {ts_idx}: {e}"
                    ))
                    var_failed += 1
                    gc.collect()

            # FLUSH this variable's images to DB immediately
            if var_records:
                try:
                    batch_size = 20
                    for i in range(0, len(var_records), batch_size):
                        batch = var_records[i:i + batch_size]
                        CachedContourImage.objects.bulk_create(batch)

                    total_size_kb = sum(len(r.image_data) for r in var_records) / 1024
                    self.stdout.write(self.style.SUCCESS(
                        f"    ✓ {var_name}: {var_rendered} images stored "
                        f"({total_size_kb:.0f} KB total)"
                    ))
                except Exception as e:
                    logger.error(f"DB insert failed for {var_name}: {e}")
                    self.stdout.write(self.style.ERROR(
                        f"    ✗ DB insert failed for {var_name}: {e}"
                    ))
                    var_failed += var_rendered
                    var_rendered = 0

            total_rendered += var_rendered
            total_failed += var_failed

            # Free all references and force GC before next variable
            del var_records
            gc.collect()

        # Final summary
        elapsed = time.time() - start_time
        if total_rendered > 0:
            self.stdout.write(self.style.SUCCESS(
                f"\n  ✓ Contour cache complete: {total_rendered} images "
                f"({total_failed} failed) in {elapsed:.0f}s"
            ))
        else:
            self.stderr.write(self.style.ERROR("  ✗ No images were rendered"))
