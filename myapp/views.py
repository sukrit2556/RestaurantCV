from django.shortcuts import render
from django.template import loader
from .models import CustomerEvents
from django.http import HttpResponse



# Create your views here.
def login(request):
    return render(request,"log-in.html")

def signup(request):
    return render(request,"signup.html")

def menu(request):
    return render(request,"menu.html")

def CustDetail(request, id):
    customer_detail = CustomerEvents.objects.get(customer_id = id)
    template = loader.get_template('CustEvent-detail.html')
    context = {
        'customer_detail': customer_detail,
    }
    return HttpResponse(template.render(context, request))
    

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