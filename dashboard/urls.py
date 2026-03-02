from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.home, name="home"),
    path("map/", views.weather_map, name="weather_map"),
    path("map/sites.json", views.map_sites_json, name="map_sites_json"),
    path("map/sites-hourly.json", views.map_sites_hourly_json, name="map_sites_hourly_json"),
    path("map/risk-grid.json", views.map_risk_grid_json, name="map_risk_grid_json"),
    path("map/contour.png", views.map_contour_image, name="map_contour_image"),
    path("map/contour-timestamps.json", views.map_contour_timestamps, name="map_contour_timestamps"),
    path("site/<int:site_id>/", views.site_detail, name="site_detail"),
    path("site/<int:site_id>/chart-data/", views.forecast_chart_data, name="chart_data"),
]
