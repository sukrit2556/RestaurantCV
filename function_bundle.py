import random
import cv2
import yaml
import numpy as np
import time
import datetime
from datetime import datetime
import random
from memory_profiler import profile

import mysql.connector


font = cv2.FONT_HERSHEY_COMPLEX

table_points = []
list_realtime_count = []

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

######### Initialize the table point [START] #########
with open('myconfig.yaml', 'r') as file:
    config = yaml.safe_load(file)
    #print(config)

# Access the points from the config dictionary
for key, value in config["table_coordinate"].items():
    print(value)
    table_points.append(value)

# Convert lists to numpy arrays
table_points = np.array(table_points, dtype=np.int32)
######### Initialize the table point [END] #########

### initialize the zero list of people count for every table
## Use for counting people
for _ in table_points:
    list_realtime_count.append(0)

def color_selector():
    with open("utils/coco.txt", "r") as my_file:
        data = my_file.read()
        class_list = data.split("\n")

    detection_colors = []
    for _ in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))

    return detection_colors, class_list

def draw_table_point(frame, availability_cache, list_point_all_table = table_points):
    ## draw and put text for each table
    for table_no, pts in enumerate(list_point_all_table):
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        ##put text on each table
        x_coord = pts[0][0]
        y_coord = pts[0][1]
        cv2.putText(frame, 
                    "Table " + str(table_no+1),
                    (x_coord,y_coord),
                    font,       #font name
                    1,          #font scale
                    (0,0,255),  #font color
                    2           #font thickness
        )
        cv2.putText(frame, 
                    "persons: " + str(list_realtime_count[table_no]),
                    (x_coord,y_coord+20),
                    font,       #font name
                    1,          #font scale
                    (0,0,255),  #font color
                    2           #font thickness
        )
        cv2.putText(frame, 
                    "realtime_status: " + str("Unoccupied" if list_realtime_count[table_no] == 0 else "occupied"),
                    (x_coord,y_coord+40),
                    font,       #font name
                    1,          #font scale
                    (0,0,255),  #font color
                    2           #font thickness
        )
        cv2.putText(frame, 
                    "Real_status: " + (str(availability_cache[table_no]) if len(availability_cache) > 0 else ""),
                    (x_coord,y_coord+60),
                    font,       #font name
                    1,          #font scale
                    (0,0,255),  #font color
                    2           #font thickness
        )

    return frame

def show_time(frame):
    current_time = datetime.datetime.now()
    print(current_time)
    cv2.putText(frame, 
                str(current_time),
                (1100,700),
                font,       #font name
                1,          #font scale
                (0,0,255),  #font color
                2           #font thickness
    )

def is_inside(edges, xp, yp): #ray-casting algorithm
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    return cnt%2 == 1

def count_table_people(horizon_center, vertical_center):
    for table_no, pts in enumerate(table_points):
        pts = pts.tolist()
        pts.append(pts[0])
        edges = list(zip(pts, pts[1:] + pts[:1]))
        if is_inside(edges, horizon_center, vertical_center):
            list_realtime_count[table_no] += 1


def reset_people_count():
    list_realtime_count.clear()
    for i in range (len(table_points)):
        list_realtime_count.append(0)
      
def put_text_bottom_right(frame, text_to_put_list):
    start_position_x = 500
    start_position_y = 650
    for text in text_to_put_list:
        cv2.putText(frame, 
                text,
                (start_position_x,start_position_y),
                font,       #font name
                1,          #font scale
                (0,0,255),  #font color
                2           #font thickness
        )
        start_position_y += 30

def connect_db():
    mydb = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="restaurant"
        )
    return mydb

def insert_db(table_name, field_list, value_list):
    mydb = connect_db()
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

def update_db(table_name, field_to_edit, new_value, condition_list):
    mydb = connect_db()
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

def delete_data_db(table_name, condition_list):
    mydb = connect_db()
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

def select_db(table_name, field_name, where_condition):
    mydb = connect_db()
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

@profile 
def do_something():
    some_list = []

    for i in range (0, 6):
        some_list.append([])
    print(some_list)
    for sublist in some_list:
        print(sublist)
        while len(sublist) < 100:
            random_number = random.randint(1, 9)
            sublist.append(random_number)

    print(some_list)


if __name__ == "__main__":
    #update_db("test", "name", "sukei", ["address = 'Highway21'", "text2 = 'suk'"])
    do_something()
    select_db("customer_events", ["*"], ["1"])
    

    