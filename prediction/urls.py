from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_charge, name='predict_charge'),
    
]
