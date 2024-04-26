from django.db import models
from taggit.managers import TaggableManager

class CustomerEvents(models.Model):
    customer_id = models.AutoField(db_column='customer_ID', primary_key=True)  # Field name made lowercase. The composite primary key (customer_ID, created_datetime) found, that is not supported. The first column is selected.
    tableid = models.IntegerField(db_column='tableID')  # Field name made lowercase.
    customer_amount = models.IntegerField()
    customer_in = models.DateTimeField(db_column='customer_IN')  # Field name made lowercase.
    customer_out = models.DateTimeField(db_column='customer_OUT')  # Field name made lowercase.
    time_getfood = models.DateTimeField(db_column='time_getFood')  # Field name made lowercase.
    captured_video = models.CharField(max_length=50)
    getfood_frame = models.CharField(max_length=50)
    created_datetime = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'customer_events'
        unique_together = (('customer_id', 'created_datetime'),)
# Create your models here.
