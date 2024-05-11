from django.shortcuts import render
from django.template import loader

from .models import CustomerEvents
from .models import SuspiciousEvents
from .models import Employee

from django.http import HttpResponse
from django.shortcuts import redirect, get_object_or_404
from django.urls import reverse

import os
from django.conf import settings
from django.utils.text import slugify
from datetime import datetime


def resolve_suspicious_event(request, sus_id):
    try:
        suspicious_event = SuspiciousEvents.objects.get(sus_id=sus_id)
        suspicious_event.sus_status = 0  # Update sus_status to 0
        suspicious_event.save()
        # Redirect to the same page after resolving the event
        return redirect('SusTable')
    except SuspiciousEvents.DoesNotExist:
        # Handle the case where the suspicious event does not exist
        return render(request, 'error_page.html', {'error_message': 'Suspicious event not found.'})


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
    customer_events = CustomerEvents.objects.all().order_by('-customer_in').values()
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
    suspicious_events = SuspiciousEvents.objects.all().order_by('-sus_datetime').values()
    context = {
        'suspicious_events': suspicious_events,
    }
    return render(request,"SusEvent-table.html",context)



#def SusTable(request):
    # Fetch suspicious events
    print("This is inside SusTable function")
    suspicious_events = SuspiciousEvents.objects.all().order_by('-sus_datetime')

    # Fetch employee names for suspicious events
    employee_names = {}
    
    for event in suspicious_events:
        print(f"suspicious event id = {event.sus_id}")
        try:
            employee = Employee.objects.get(employee_id=event.sus_employeeid)
            print(event.sus_employeeid)
            employee_names[event.sus_employeeid] = f"{employee.employee_name} {employee.employee_sname}"
        except Employee.DoesNotExist:
            print("get into except")
            # Handle case where employee does not exist
            employee_names[event.sus_employeeid] = "Unknown Employee"
    print(f"employee_names = {employee_names}")
    # Pass the suspicious events and employee names to the template
    return render(request, 'SusEvent-table.html', 
                  {'suspicious_events': suspicious_events,
                   'employee_names': employee_names})






def ResTable(request):
    suspicious_events = SuspiciousEvents.objects.all().order_by('-sus_datetime').values()
    context = {
        'suspicious_events': suspicious_events,
    }
    return render(request,"ResEvent-table.html",context)

def ListTable(request):
    employee_table = Employee.objects.all()
    context = {
        'employee_table': employee_table
    }
    return render(request,"ListEm-table.html",context)

def AddEm(request):
    if request.method == 'POST':
        # Retrieve data from the form
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        photo = request.FILES.get('photo')
        
        # Generate a unique filename for the photo
        new_filename = generate_unique_filename(photo.name)
        
        # Handle file upload
        handle_uploaded_file(photo, new_filename)
        
        # Save the photo with the /mock_media/ path in the database
        photo_path = os.path.join('/mock_media/', new_filename)
        
        # Create a new Employee object and save it to the database
        new_employee = Employee.objects.create(
            employee_name=first_name,
            employee_sname=last_name,
            employee_image=photo_path
        )
        
        # Redirect to the list of employees
        return redirect('ListTable')
    else:
        # Render the Add Employee form template for GET requests
        return render(request, 'AddEm.html')


#def EditEm(request, id):
#    employee_detail = Employee.objects.get(employee_id=id)
#    context = {
#            'employee_detail': {
#            'employee_name': employee_detail.employee_name,
#            'employee_sname': employee_detail.employee_sname,
#            'employee_image': employee_detail.employee_image,
#            'employee_id': employee_detail.employee_id
#            }
#        }
#    return render(request, 'EditEm.html', context)

def handle_uploaded_file(file, filename):
    # Define the directory where you want to save the uploaded files
    upload_dir = os.path.join(settings.MEDIA_ROOT, '')

    # Create the directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)

    # Join the directory path and the filename to get the full file path
    file_path = os.path.join(upload_dir, filename)

    # Open the file and write the uploaded content to the destination file
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

def generate_unique_filename(original_filename):
    # Split the original filename into base and extension
    base, extension = os.path.splitext(original_filename)
    # Generate a unique filename using current timestamp and original file extension
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Append original filename and extension to the timestamp
    new_filename = f"{timestamp}_{slugify(base)}{extension}"
    return new_filename

def EditEm(request, id):
    employee_detail = Employee.objects.get(employee_id=id)

    if request.method == 'POST':
        # Update employee data based on form submission
        employee_detail.employee_name = request.POST.get('first_name')
        employee_detail.employee_sname = request.POST.get('last_name')
        
        # Handle image file upload
        if 'photo' in request.FILES:
            photo_file = request.FILES['photo']

            new_filename = generate_unique_filename(photo_file.name)
            handle_uploaded_file(photo_file, new_filename)
            # Update employee_detail with the new filename
            employee_detail.employee_image = os.path.join('\mock_media', new_filename)
        
        # Save the updated employee object
        employee_detail.save()
        
        return redirect('EditEm', id=employee_detail.employee_id)  # Redirect to employee detail page
    else:
        # Handle GET request
        return render(request, 'EditEm.html', {'employee_detail': employee_detail})
    
def delete_employee(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    employee.delete()
    return redirect('ListTable')

def Dashboard(request):
    customer_events = CustomerEvents.objects.all()
    context = {
        'customer_events': customer_events
    }
    return render(request,"Dashboard.html",context)