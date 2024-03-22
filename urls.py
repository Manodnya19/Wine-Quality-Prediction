from django.urls import path

from . import views

urlpatterns = [

    path('index/', views.index, name ='index'),
    path('predict/', views.predict, name='predict'),
    path('foodandwine/', views.foodandwine, name='foodandwine'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact')

]
