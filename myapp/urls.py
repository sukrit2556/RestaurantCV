from django.urls import path
from myapp import views

urlpatterns = [
    path('login',views.login),
    path('signup',views.signup),
    path('menu/',views.menu, name='menu'),
    #path('CustDetail',views.CustDetail),
    #path('CustTable/',views.CustTable),
    #path('EmTable',views.EmTable),
    #path('SusTable',views.SusTable),
    #path('ResTable',views.ResTable),
    #path('ListTable',views.ListTable),
    #path('Dashboard',views.Dashboard),
    #path('CustTable/', views.CustTable, name='CustTable'),
    path('menu/CustTable/CustDetail/<int:id>', views.CustDetail, name='details'),
    path('menu/CustTable/', views.CustTable, name='CustTable'),
    path('menu/EmTable/', views.EmTable, name='EmTable'),
    path('menu/SusTable/', views.SusTable, name='SusTable'),
    path('menu/ResTable/', views.ResTable, name='ResTable'),
    path('menu/ListTable/', views.ListTable, name='ListTable'),
    path('menu/ListTable/AddEM/', views.AddEm, name='AddEm'),
    path('menu/ListTable/EditEm/<int:id>', views.EditEm, name='EditEm'),
    path('delete_employee/<int:employee_id>/', views.delete_employee, name='delete_employee'),
    path('menu/ListTable/AddEM/', views.AddEm, name='AddEm'),
     path('resolve_suspicious_event/<int:sus_id>/', views.resolve_suspicious_event, name='resolve_suspicious_event'),
    ]