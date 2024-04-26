from django.shortcuts import render
from .models import CustomerEvents


# Create your views here.
def login(request):
    return render(request,"log-in.html")

def signup(request):
    return render(request,"signup.html")

def menu(request):
    return render(request,"menu.html")

def CustDetail(request):
    return render(request,"CustEvent-detail.html")

def CustTable(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"CustEvent-table.html",context)

def EmTable(request):
    return render(request,"EmEvent-table.html")

def SusTable(request):
    return render(request,"SusEvent-table.html")

def ResTable(request):
    return render(request,"ResEvent-table.html")

def ListTable(request):
    return render(request,"ListEm-table.html")

def Dashboard(request):
    return render(request,"Dashboard.html")