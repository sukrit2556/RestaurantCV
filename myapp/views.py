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

    customer_in = customer_detail.customer_in
    customer_out = customer_detail.customer_out
    time_getFood = customer_detail.time_getfood

    if customer_out:
        time_spent = customer_out - customer_in

        # Convert time_spent to hours and minutes
        hours, remainder = divmod(time_spent.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        # Format the output
        time_spent_formatted = f"{hours} hours and {minutes} minutes"
    else:
        # If customer_out is None, set time_spent_formatted to a default value
        time_spent_formatted = "Customer still present"

    # Pass the formatted time spent to the template context
    if customer_in and time_getFood:
        # Calculate the waiting time for food
        waiting_time_for_food = time_getFood - customer_in

        # Convert waiting_time_for_food to minutes
        waiting_time_minutes = waiting_time_for_food.seconds // 60

        # Format the output
        waiting_time_formatted = f"{waiting_time_minutes} minutes"
    else:
        waiting_time_formatted = "N/A"  # Default value if either customer_in or time_getFood is None
    context = {
        'customer_detail': customer_detail,
        'time_spent_formatted': time_spent_formatted,
        'waiting_time_formatted': waiting_time_formatted
    }
    return render(request, 'CustEvent-detail.html', context)


#def back_to_cust_table(request):
    # Generate the URL for the CustTable view
    cust_table_url = reverse('CustTable')
    # Redirect the user to the CustTable page
    return redirect(cust_table_url)

  
def CustTable(request):
    customer_events = CustomerEvents.objects.all().order_by('-customer_in').values()
    print(customer_events)
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
        photo_path = os.path.join('/mock_media/employee_face', new_filename)
        
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
            employee_detail.employee_image = os.path.join('\mock_media\employee_face', new_filename)
        
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

from django.db.models import Sum, Count
from django.db.models.functions import ExtractMonth, ExtractYear
import calendar
from django.db.models import F, ExpressionWrapper, DurationField
from django.utils import timezone
from datetime import timedelta
from django.db.models import Avg
from django.db.models.functions import ExtractDay
from django.db.models import Avg, ExpressionWrapper, fields, F, DurationField, Sum
from django.db.models.functions import TruncDate
from datetime import timedelta


def Dashboard(request):
    # Fetching data from the CustomerEvents table and aggregating the total customer amount per month
    customers = CustomerEvents.objects.annotate(
        month=ExtractMonth('customer_in'),
        year=ExtractYear('customer_in')
    ).values('month', 'year').annotate(
        total_customers=Sum('customer_amount')
    )

    # Fetching data for the count of records for each sus_type (0 for drawer type and 1 for employee type)
    sus_type_counts = SuspiciousEvents.objects.values('sus_type').annotate(count=Count('*'))

    # Extracting the count for each sus_type
    drawer_type_count = 0
    employee_type_count = 0
    for item in sus_type_counts:
        if item['sus_type'] == 0:
            drawer_type_count = item['count']
        elif item['sus_type'] == 1:
            employee_type_count = item['count']

    # Fetching data for the count of records for each sus_status (resolved and unresolved)
    sus_status_counts = SuspiciousEvents.objects.values('sus_status').annotate(count=Count('*'))

    # Extracting the count for resolved and unresolved events
    resolved_count = unresolved_count = 0
    for item in sus_status_counts:
        if item['sus_status'] == 0:
            resolved_count = item['count']
        elif item['sus_status'] == 1:
            unresolved_count = item['count']

    customer_events = CustomerEvents.objects.all()

    # Calculate waiting time for each event and group by date
    waiting_data = {}  # Dictionary to store waiting time per day
    for event in customer_events:
        day = event.customer_in.date()  # Get the date of the event
        if event.time_getfood:  # Check if food was served
            waiting_time = event.time_getfood - event.customer_in  # Calculate waiting time
            if day in waiting_data:
                waiting_data[day].append(waiting_time.total_seconds() / 60)  # Convert to minutes
            else:
                waiting_data[day] = [waiting_time.total_seconds() / 60]  # Convert to minutes

    # Calculate average waiting time for each day
    avg_waiting_per_day = {}
    for day, waiting_times in waiting_data.items():
        avg_waiting_per_day[day] = sum(waiting_times) / len(waiting_times)

    # Sort the dictionary by date if needed
    avg_waiting_per_day = dict(sorted(avg_waiting_per_day.items()))

    # Prepare data for passing to the template
    waiting_labels = list(avg_waiting_per_day.keys())
    waiting_data = list(avg_waiting_per_day.values())

    # Extracting data for the line graph
    labels = [calendar.month_name[entry['month']] + ' ' + str(entry['year']) for entry in customers]
    data = [entry['total_customers'] for entry in customers]

    # Extracting data for the top months with the highest service usage
    top_months = CustomerEvents.objects.annotate(
        month=ExtractMonth('customer_in')
    ).values('month').annotate(
        total_customerss=Sum('customer_amount')
    ).order_by('-total_customerss')[:3]

    context = {
        'labels': labels,
        'data': data,
        'top_months': top_months,
        'drawer_count': drawer_type_count,
        'employee_count': employee_type_count,
        'resolved_count': resolved_count,
        'unresolved_count': unresolved_count,
        'waiting_labels': waiting_labels,
        'waiting_data': waiting_data,
    }

    return render(request, 'Dashboard.html', context)