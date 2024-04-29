from django.shortcuts import render
from django.template import loader
from .models import CustomerEvents
from django.http import HttpResponse
from django.shortcuts import redirect
from django.urls import reverse


# Create your views here.
def login(request):
    return render(request,"log-in.html")

def signup(request):
    return render(request,"signup.html")

#def menu(request):
    # Redirect to the CustTable URL
    return redirect('CustTable')#

def menu(request):
    return render(request,"menu.html")

#def CustDetail(request, id):
    customer_detail = CustomerEvents.objects.get(customer_id = id)
    template = loader.get_template('CustEvent-detail.html')
    context = {
        'customer_detail': customer_detail,
    }
    return HttpResponse(template.render(context, request))

def CustDetail(request, id):
    customer_detail = CustomerEvents.objects.get(customer_id=id)
    context = {
        'customer_detail': customer_detail,
    }
    return render(request, 'CustEvent-detail.html', context)


#def back_to_cust_table(request):
    # Generate the URL for the CustTable view
    cust_table_url = reverse('CustTable')
    # Redirect the user to the CustTable page
    return redirect(cust_table_url)

  
def CustTable(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"CustEvent-table.html",context)

def EmTable(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"EmEvent-table.html",context)

def SusTable(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"SusEvent-table.html",context)

def ResTable(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"ResEvent-table.html",context)

def ListTable(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"ListEm-table.html",context)

def Dashboard(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"Dashboard.html",context)