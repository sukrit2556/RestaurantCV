from django.db import models
from taggit.managers import TaggableManager

class CustomerEvents(models.Model):
    customer_id = models.AutoField(db_column='customer_ID', primary_key=True)  # Field name made lowercase. The composite primary key (customer_ID, created_datetime) found, that is not supported. The first column is selected.
    tableid = models.IntegerField(db_column='tableID')  # Field name made lowercase.
    customer_amount = models.IntegerField()
    customer_in = models.DateTimeField(db_column='customer_IN')  # Field name made lowercase.
    customer_out = models.DateTimeField(db_column='customer_OUT')  # Field name made lowercase.
    time_getfood = models.DateTimeField(db_column='time_getFood')  # Field name made lowercase.
    captured_video = models.CharField(max_length=200)
    getfood_frame = models.CharField(max_length=200)
    created_datetime = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'customer_events'
        unique_together = (('customer_id', 'created_datetime'),)

class Employee(models.Model):
    employee_id = models.AutoField(db_column='employee_ID', primary_key=True)  # Field name made lowercase.
    employee_name = models.CharField(max_length=250)
    employee_sname = models.CharField(max_length=250)
    employee_image = models.CharField(max_length=250)

    class Meta:
        managed = False
        db_table = 'employee'


class SuspiciousEvents(models.Model):
    sus_id = models.AutoField(db_column='sus_ID', primary_key=True)  # Field name made lowercase. The composite primary key (sus_ID, sus_datetime) found, that is not supported. The first column is selected.
    sus_type = models.IntegerField(db_comment='0 = drawer, 1 = employee')
    sus_employeeid = models.IntegerField(db_column='sus_employeeID')  # Field name made lowercase.
    sus_video = models.CharField(max_length=250)
    sus_status = models.IntegerField(db_comment='0 = resolved, 1 = not resolved')
    sus_datetime = models.DateTimeField()
    sus_where = models.IntegerField(db_comment='0 = cashier, 1-6 = table 1-6')

    class Meta:
        managed = False
        db_table = 'suspicious_events'
        unique_together = (('sus_id', 'sus_datetime'),)


class TableList(models.Model):
    tableid = models.IntegerField(db_column='tableID', primary_key=True)  # Field name made lowercase.
    table_status = models.IntegerField(db_comment='0 = unoccupied\r\n1 = occupied\r\n2 = unoccupied dirty')

    class Meta:
        managed = False
        db_table = 'table_list'

# Create your models here.
