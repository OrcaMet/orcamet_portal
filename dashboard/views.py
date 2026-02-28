"""
OrcaMet Portal — Dashboard Views

The main views a logged-in user sees.
Phase 1: Simple welcome page with site list.
Phase 2: Full forecast display, maps, etc.
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from sites.models import Site


@login_required(login_url="/login/")
def home(request):
    """
    Main dashboard view.

    Superadmins see all sites.
    Client users see only their client's sites.
    """
    user = request.user

    if user.is_superadmin:
        sites_list = Site.objects.filter(is_active=True).select_related("client")
    elif user.client:
        sites_list = Site.objects.filter(
            client=user.client, is_active=True
        ).select_related("client")
    else:
        sites_list = Site.objects.none()

    context = {
        "user": user,
        "sites": sites_list,
        "site_count": sites_list.count(),
    }

    return render(request, "dashboard/home.html", context)
