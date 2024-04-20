import datetime
import time
import cv2
from datetime import datetime
import traceback
from ultralytics import YOLO
import statistics
import threading

import mysql.connector

class Mythread(threading.Thread):
    def __init__(self, table_points):
        super().__init__()
        self.table_points = table_points
        self.object1 = videoQueue
        self.stop_event = threading.Event()
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()

    def calculate_real_people_total(self):
        global list_total_count_cache, stop_thread
        #initialize sampling list
        sampling_from_tables = []
        table_name = "customer_events"
        for _ in range (len(self.table_points)):
            sampling_from_tables.append([])

        try:
            while not stop_thread:
                if check_available_started:
                    print("________________-Get in calculate now")
                    list_total_count = list_realtime_count_cache.copy()
                    
                    print("sampling_from_tables", sampling_from_tables)

                    # operating
                    for i in range (len(self.table_points)):

                        print(i)
                    
                        # if occupied then keep collecting until it reaches 100 collection
                        if availability_cache[i] == "occupied" and len(sampling_from_tables[i]) < 100:
                            realtime_count_that_table = list_total_count[i]
                            sampling_from_tables[i].append(realtime_count_that_table)

                        elif availability_cache[i] == "unoccupied": #if become unoccupied then reset
                            sampling_from_tables[i].clear()
                            
                        
                    print("helloooooooooooooooooooooooooooooooooooooooooooooooooooo")
                    print(sampling_from_tables)

                    # determine the meaning
                    for i in range (len(self.table_points)):
                        if len(sampling_from_tables[i]) > 0: # have sampling data
                            list_total_count[i] = statistics.mode(sampling_from_tables[i])
                            #list_total_count[i] = max(sampling_from_tables[i])
                            #list_total_count[i] = round(sum(sampling_from_tables[i])/len(sampling_from_tables[i]))
                            if (len(sampling_from_tables[i]) == 1 or len(sampling_from_tables[i]) == 25 or len(sampling_from_tables[i]) == 50 or 
                            len(sampling_from_tables[i]) == 75 or len(sampling_from_tables[i]) == 100):
                                #UPDATE customer_events SET customer_amount = %s WHERE 
                                #customer_IN = (SELECT MAX(customer_IN) from customer_events WHERE tableID = %s)
                                self.update_db(table_name, "customer_amount", list_total_count[i], 
                                        ["customer_IN = (" + self.select_db("customer_events", ["MAX(customer_IN)"], [f"tableID = {i+1}"]) + ")"])
                                
                        else: # have no sampling data
                            list_total_count[i] = 0
                    list_total_count_cache = list_total_count.copy()
                    list_total_count.clear()

                    time.sleep(5)
        except Exception as e:
            print("error: ", e)
            traceback.print_exc()
            stop_thread = True

    def check_available(self):
        global list_realtime_count_cache, stop_thread, availability_cache, check_available_started
        table_name = "customer_events"
        while not stop_thread:
            table_status = [] # 3 = occupied , 0 = unoccupied
            availability = []
            try:
                for table_no, _ in enumerate(self.table_points): #initialize list
                    table_status.append(0)
                # start checking for 3 sampling time at each tables
                for i in range (0, 5):
                    print("check available ====================================================", i)
                    list_realtime_count = list_realtime_count_cache.copy()
                    for table_no, _ in enumerate(self.table_points):
                        if list_realtime_count[table_no] > 0: #occupied detected that moment
                            table_status[table_no] += 1
                    time.sleep(4)

                print("table_status", table_status)

                # determining the meaning of state
                for i, item in enumerate(table_status):
                    if item >= 5/2:   # occupied
                        availability.append("occupied")
                        if len(availability_cache) == 0 or availability_cache[i] == "unoccupied":
                            field_list = ["tableID", "customer_IN"]
                            value_list = [i+1, datetime.now()]
                            self.insert_db(table_name, field_list, value_list)
                    else:                   #unoccupied
                        availability.append("unoccupied")
                        print(availability)
                        if len(availability_cache) != 0 and availability_cache[i] == "occupied":
                            print("facckkkkk")
                            self.update_db(table_name, "customer_OUT", datetime.now(), 
                                    ["customer_IN = (" + self.select_db("customer_events", ["MAX(customer_IN)"], [f"tableID = {i+1}"]) + ")"])
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                print(availability)

                availability_cache = availability.copy()
                if check_available_started == False:
                    check_available_started = True
                

            except Exception as e:
                print("error: ", e)
                traceback.print_exc()
                stop_thread = True

    def now_frame_rate(self):
        global frame_rate
        global frame_count 
        second = 1
        while not stop_thread:
            with self.lock:
                frame_count_before = frame_count
            time.sleep(second)
            frame_count_after = frame_count
            frame_rate = int((frame_count_after - frame_count_before)/second)

    def combine_frame(self):
        completed_frames = None
        global stop_thread
        # Define video resolution and frame rate
        frame_width = 1920
        frame_height = 1080
        fps = 30

        size = (frame_width, frame_height) 
        # Define codec and create VideoWriter object
        out = cv2.VideoWriter('result_video/TestOccupied.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            15, size) 

        while not stop_thread and not end_recording:
            print("COMBINE FRAME IS STILL WORKING FINE")
            # Iterate over each video queue to check for frames
            if completed_frames is None:  # Check if frame for this index is not yet filled
                frame = self.object1.get_frame()  # Get frame from video queue
                
                if frame is not None:  # If frame is available, update completed_frames
                    completed_frames = frame
                    
            if frame is not None:
                print(completed_frames)
                completed_frames = cv2.resize(completed_frames, (1920,1080))
                out.write(completed_frames)
                # Display the combined frame
                #cv2.imshow('completed_frames', completed_frames)
                completed_frames = None
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            #time.sleep(0.1)
        out.release()

    def connect_db(self):
        mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="restaurant"
            )
        return mydb

    def insert_db(self, table_name, field_list, value_list):
        mydb = self.connect_db()
        mycursor = mydb.cursor()

        # Construct placeholders for values
        field_placeholders = ', '.join(['%s' for _ in field_list])

        # Construct the SQL query
        sql = f"INSERT INTO {table_name} ({', '.join(field_list)}) VALUES ({field_placeholders})"

        # Execute the query
        mycursor.execute(sql, value_list)

        # Commit the transaction
        mydb.commit()

        print(mycursor.rowcount, "record inserted.")
        mycursor.close()
        mydb.close()

    def update_db(self, table_name, field_to_edit, new_value, condition_list):
        mydb = self.connect_db()
        mycursor = mydb.cursor()

        # Create placeholders for conditions
        condition_placeholders = ' AND '.join(['{}'.format(condition) for condition in condition_list])
        print(condition_placeholders)

        # Construct the SQL query string with placeholders
        sql = f"UPDATE {table_name} SET {field_to_edit} = %s WHERE {condition_placeholders}"


        # Print the SQL query (for debugging purposes)
        print("Generated SQL query:", sql)

        # Execute the query
        mycursor.execute(sql, [new_value])

        # Commit the transaction
        mydb.commit()

        print(mycursor.rowcount, "record(s) updated.")
        mycursor.close()
        mydb.close()

    def delete_data_db(self, table_name, condition_list):
        mydb = self.connect_db()
        mycursor = mydb.cursor()

        # Create placeholders for conditions
        condition_placeholders = ' AND '.join(['{}'.format(condition) for condition in condition_list])
        print(condition_placeholders)

        # Construct the SQL query string with placeholders
        sql = f"DELETE FROM {table_name} WHERE {condition_placeholders}"


        # Print the SQL query (for debugging purposes)
        print("Generated SQL query:", sql)

        # Execute the query
        mycursor.execute(sql)

        # Commit the transaction
        mydb.commit()

        print(mycursor.rowcount, "record(s) deleted.")
        mycursor.close()
        mydb.close()

    def select_db(self, table_name, field_name, where_condition):
        mydb = self.connect_db()
        mycursor = mydb.cursor()

        formated_field_name = ', '.join(['{}'.format(field_names) for field_names in field_name])
        formated_condition = ', '.join(['{}'.format(where_conditions) for where_conditions in where_condition])
        # Construct the SQL query string with placeholders
        sql = f"SELECT {formated_field_name} from {table_name} WHERE {formated_condition}"
        print("generated sql = ", sql)

        mycursor.execute(sql)

        myresult = mycursor.fetchall()

        for result in myresult:
            print(result)
        
        mycursor.close()
        mydb.close()

        return sql # for subquery uses


class videoQueue:
    def __init__(self):
        self.frame_queue = []
        
    def add_frame(self, frame):
        self.frame_queue.append(frame)
        
    def get_frame(self):
        if len(self.frame_queue) > 0:
            return self.frame_queue.pop(0)
        return None
    
    def is_empty(self):
        if len(self.frame_queue) == 0:
            return True
        else:
            return False