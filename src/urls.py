from django.urls import path, include

urlpatterns = [
    path('api/', include('dme_selector.urls')),
]