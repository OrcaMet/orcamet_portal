"""
OrcaMet Portal — URL Configuration
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("accounts.urls")),
    path("dashboard/", include("dashboard.urls")),
    path("sites/", include("sites.urls")),
    path("forecasts/", include("forecasts.urls")),
]
