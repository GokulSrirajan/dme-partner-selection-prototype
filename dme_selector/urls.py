from django.urls import path
from .views import select_dme_partner

urlpatterns = [
    path('select-dme/', select_dme_partner),
]
