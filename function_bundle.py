import random
import cv2
import yaml
import numpy as np
import time
import datetime

import mysql.connector

mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="",
  database="restaurant"
)

font = cv2.FONT_HERSHEY_COMPLEX

table_points = []
list_realtime_count = []



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
    for i in range (len(table_points)):
        list_realtime_count[i] = 0
      
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

def insert_db(table_name, field_list, value_list):
    mycursor = mydb.cursor()

    table_name = table_name

    # Construct placeholders for values
    field_placeholders = ', '.join(['%s' for _ in field_list])

    # Construct the SQL query
    sql = f"INSERT INTO {table_name} ({', '.join(field_list)}) VALUES ({field_placeholders})"

    # Execute the query
    mycursor.execute(sql, value_list)

    # Commit the transaction
    mydb.commit()

    print(mycursor.rowcount, "record inserted.")

if __name__ == "__main__":
    insert_db("test", ["name", "address", "text1", "text2", "text3"], ["John", "Highway21", "fuck", "dsd", "ssss"])