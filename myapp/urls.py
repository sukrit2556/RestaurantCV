from django.urls import path
from myapp import views

urlpatterns = [
    path('login',views.login),
    path('signup',views.signup),
    path('menu',views.menu),
    path('CustDetail',views.CustDetail),
    path('CustTable',views.CustTable),
    path('EmTable',views.EmTable),
    path('SusTable',views.SusTable),
    path('ResTable',views.ResTable),
    path('ListTable',views.ListTable),
    path('Dashboard',views.Dashboard),
]