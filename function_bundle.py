import random
import cv2
import yaml
import numpy as np
import time
import datetime
from datetime import datetime, timedelta
import random
from ultralytics import YOLO
import os
import traceback
from database_action import *

font = cv2.FONT_HERSHEY_COMPLEX

table_points = []
table_crop_points = []
list_realtime_count = []
count_person_at_table = []
realtime_dimsum_found = []
to_check = []
fps = 0
stop_dimsum_thread = [False for _ in range (0,6)]
blank_frame_cache = None
human_dict = {}

class videoQueue:   # for recording in thread 4
    def __init__(self):
        self.frame_queue = []
        self.date_time_queue = []
        
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
    
    def get_len(self):
        return len(self.frame_queue)
    
    def clear_all(self):
        self.frame_queue.clear()
        self.date_time_queue.clear()

    def add_frame_obj(self, frame_obj):
        self.frame_queue.append(frame_obj.frame)
        self.date_time_queue.append(frame_obj.date_time)

    def get_frame_obj(self):
        if len(self.frame_queue) > 0:
            obj = frame_attr(self.frame_queue.pop(0), self.date_time_queue.pop(0))
            return obj
        return None


class frame_attr():
    def __init__(self, frame, date_time):
        self.frame = frame
        self.date_time = date_time

class person():
    def __init__(self, person_type, frame_first_found, frame_latest_found, probToBeCustomer,
                 dt_first_found, dt_latest_found, fixed, top_left, bottom_right):
        self.person_type = person_type
        self.frame_first_found = frame_first_found
        self.frame_latest_found = frame_latest_found
        self.probToBeCustomer = probToBeCustomer
        self.dt_first_found = dt_first_found
        self.dt_latest_found = dt_latest_found
        self.fixed = fixed
        self.pixel_list = []
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.saved_frame = []
        self.saved_frame_latest_dt = present_datetime

    def add_pixel(self, coordinate:tuple):
        """
        add pixel to add trace of walking
        arg = coordinate in (x, y)
        return nothing
        """
        if len(self.pixel_list) <= 20:
            self.pixel_list.append(coordinate)
        else:
            self.pixel_list.pop(0)
            self.pixel_list.append(coordinate)

    def all_edge(self):
        edge_list = []
        for i in range (len(self.pixel_list) - 1):
            point1 = self.pixel_list[i]
            point2 = self.pixel_list[i+1]
            edge_list.append([point1, point2])
        return edge_list
    

######### Initialize the table point [START] #########
with open('myconfig.yaml', 'r') as file:
    config = yaml.safe_load(file)

        ### load a pretrained YOLOv8n model ###
    model = YOLO(config['main_model_path'])

    ### select the source of captured frame ###
    source = config['source']
    if source == "video_frame":
        url_path = config['video_frame']['url']
        cap = cv2.VideoCapture(url_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if config['video_frame']['simulate'] == True:
            simulate = True
        else:
            simulate = False

    elif source == "live_frame":
        url_path = config['live_frame']['url']
        cap = cv2.VideoCapture(url_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

    ### Amount of frame skipped ####
    frame_skipped = config['frame_skip']

    ##  color of person type
    person_color = [tuple(config['colors']['unknown']), tuple(config['colors']['customer']), tuple(config['colors']['employee'])]

    if config['datetime'] == "now":
        start_datetime = datetime.now()
    else:
        start_datetime = config['datetime']

# Access the points from the config dictionary
for key, value in config["table_coordinate"].items():
    table_points.append(value)

# Convert lists to numpy arrays
table_points = np.array(table_points, dtype=np.int32)


# Access the points from the config dictionary
for key, value in config["table_crop_coord"].items():
    table_crop_points.append(value)

# Convert lists to numpy arrays
table_crop_points = np.array(table_crop_points, dtype=np.int32)
######### Initialize the table point [END] #########

### initialize the zero list of people count for every table
## Use for counting people
for _ in table_points:
    list_realtime_count.append(0)

for _ in table_points:
    count_person_at_table.append(0)

# use for set status what to check dimsum
for _ in table_points:
    to_check.append(0)

for _ in table_points:
    realtime_dimsum_found.append(0)

object1 = videoQueue() #used for thread4 (recording)
object2 = [] #used for detect dimsum
check_dimsum_thread_list = []
present_datetime = start_datetime
for i in range (len(table_points)):
    obj2 = videoQueue()
    object2.append(obj2)
    check_dimsum_thread_list.append(0)

#for recording customer
plotted_points = []
record_width_height = []
for key, value in  config['table_coord_for_recording'].items():
    top_left = [value[0][0], value[0][1]]
    top_right = [value[1][0], value[0][1]]
    bottom_right = [value[1][0], value[1][1]]
    bottom_left = [value[0][0], value[1][1]]
    full_shape = [top_left, top_right, bottom_right, bottom_left]
    width = value[1][0] - value[0][0]
    height = value[1][1] - value[0][1]
    record_width_height.append((width, height))
    plotted_points.append(full_shape)


#for detecting drawer
drawer_detect_points = []
for key, value in  config['drawer_coordination'].items():
    top_left = [value[0][0], value[0][1]]
    top_right = [value[1][0], value[0][1]]
    bottom_right = [value[1][0], value[1][1]]
    bottom_left = [value[0][0], value[1][1]]
    full_shape = [top_left, top_right, bottom_right, bottom_left]
    drawer_detect_points.append(full_shape)

#for detecting person in cashier
cashier_area_points = []
for key, value in  config['cashier_detect_area'].items():
    top_left = [value[0][0], value[0][1]]
    top_right = [value[1][0], value[0][1]]
    bottom_right = [value[1][0], value[1][1]]
    bottom_left = [value[0][0], value[1][1]]
    full_shape = [top_left, top_right, bottom_right, bottom_left]
    cashier_area_points.append(full_shape)

#for recording cashier area
cashier_area_record = []
cashier_record_width_height = []
for key, value in  config['cashier_record_area'].items():
    top_left = [value[0][0], value[0][1]]
    top_right = [value[1][0], value[0][1]]
    bottom_right = [value[1][0], value[1][1]]
    bottom_left = [value[0][0], value[1][1]]
    full_shape = [top_left, top_right, bottom_right, bottom_left]
    width = value[1][0] - value[0][0]
    height = value[1][1] - value[0][1]
    cashier_record_width_height.append((width, height))
    cashier_area_record.append(full_shape)

#for recording cashier area
employee_detect_area_points = []

# Access the points from the config dictionary
for key, value in config["employee_detect_area"].items():
    employee_detect_area_points.append(value)

# Convert lists to numpy arrays
employee_detect_area_points = np.array(employee_detect_area_points, dtype=np.int32)

#for recording employee area
employee_record_area_point = []
employee_record_width_height = []
for key, value in  config['employee_record_area'].items():
    top_left = [value[0][0], value[0][1]]
    top_right = [value[1][0], value[0][1]]
    bottom_right = [value[1][0], value[1][1]]
    bottom_left = [value[0][0], value[1][1]]
    full_shape = [top_left, top_right, bottom_right, bottom_left]
    width = value[1][0] - value[0][0]
    height = value[1][1] - value[0][1]
    employee_record_width_height.append((width, height))
    employee_record_area_point.append(full_shape)

plotted_points_recording = np.array(plotted_points, dtype=np.int32)
drawer_detect_points = np.array(drawer_detect_points, dtype=np.int32)
cashier_area_points = np.array(cashier_area_points, dtype=np.int32)
cashier_area_record = np.array(cashier_area_record, dtype=np.int32)
employee_detect_area_points = np.array(employee_detect_area_points, dtype=np.int32)
employee_record_area_point = np.array(employee_record_area_point, dtype=np.int32)
save_preview = config['save_preview']










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

#### Initialize the color randomizer for detected box ####
detection_colors, class_list = color_selector()

def draw_table_point(frame, availability_cache, list_point_all_table = table_points):
    ## draw and put text for each table
    for table_no, pts in enumerate(list_point_all_table):
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        #find leftest point
        # Find the index of the point with the minimum x-coordinate (using axis=0 for row-wise comparison)
        leftmost_index = np.argmin(pts[:, 0])

        # Extract the leftmost point
        leftmost_point = pts[leftmost_index]
        ##put text on each table
        x_coord = leftmost_point[0]
        y_coord = leftmost_point[1]
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, 
                    "Table " + str(table_no+1),
                    (x_coord,y_coord),
                    font,       #font name
                    1.5,          #font scale
                    (0,0,255),  #font color
                    2,          #font thickness
                    cv2.LINE_AA
        )
        cv2.putText(frame, 
                    "persons: " + str(list_realtime_count[table_no]),
                    (x_coord,y_coord+20),
                    font,       #font name
                    1.5,          #font scale
                    (0,0,255),  #font color
                    2,          #font thickness
                    cv2.LINE_AA
        )
        cv2.putText(frame, 
                    "RT_status: " + str("Unoccupied" if list_realtime_count[table_no] == 0 else "occupied"),
                    (x_coord,y_coord+40),
                    font,       #font name
                    1.5,          #font scale
                    (0,0,255),  #font color
                    2,          #font thickness
                    cv2.LINE_AA
        )
        cv2.putText(frame, 
                    "Real_status: " + (str(availability_cache[table_no]) if len(availability_cache) > 0 else ""),
                    (x_coord,y_coord+60),
                    font,       #font name
                    1.5,          #font scale
                    (0,0,255),  #font color
                    2,          #font thickness
                    cv2.LINE_AA
        )

    return frame

def show_time(frame):
    current_time = datetime.datetime.now()
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
            return True
    return False

def check_person_at_cashier(horizon_center, vertical_center):
    pts = drawer_detect_points[0]
    pts = pts.tolist()
    pts.append(pts[0])
    edges = list(zip(pts, pts[1:] + pts[:1]))
    if is_inside(edges, horizon_center, vertical_center):
        return True
    return False

def check_employee_at_table(horizon_center, vertical_center):
    for table_no, pts in enumerate(employee_detect_area_points):
        pts = pts.tolist()
        pts.append(pts[0])
        edges = list(zip(pts, pts[1:] + pts[:1]))
        if is_inside(edges, horizon_center, vertical_center):
            count_person_at_table[table_no] += 1
            return True, table_no
    return False, None

def reset_people_count():
    list_realtime_count.clear()
    for i in range (len(table_points)):
        list_realtime_count.append(0)

def reset_employee_at_table_count():
    count_person_at_table.clear()
    for i in range (len(table_points)):
        count_person_at_table.append(0)
      
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

def draw_from_points(frame, list_point_all_table, color:tuple):
    ## draw and put text for each table
    for table_no, pts in enumerate(list_point_all_table):
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    return frame

def add_jpg_media(table_id, filename, media_to_add):
    _, result = select_db("customer_events", ["MAX(customer_ID)"], [f"tableID = {table_id+1}"])
    #save to real dir
    path_to_save = ".." + config['save_customer_path_parent'] + config['save_customer_path_child']
    media_directory = os.path.join(os.getcwd(), path_to_save)
    new_folder_path = os.path.normpath(os.path.join(media_directory, str(result[0][0])))
    os.makedirs(new_folder_path, exist_ok=True)
    full_image_file_path = os.path.join(new_folder_path, filename)
    cv2.imwrite(full_image_file_path, media_to_add)

    #save to Database
    relative_image_file_path = os.path.normpath(os.path.join(config['save_customer_path_child'], str(result[0][0]), filename))
    retries = 0
    max_retries = 3
    ## prevent database deadlock problem
    while retries < max_retries:
        try:
            update_db("customer_events", "getfood_frame", relative_image_file_path, 
                      ["customer_ID = (" + f"{select_db('customer_events', ['MAX(customer_ID)'], [f'tableID = {table_id+1}'])[0]})", 
                       f"tableID = {table_id+1}"], verbose=True)
            break  # If successful, exit the loop
        except mysql.connector.errors.InternalError as e:
            # Check if the error is a deadlock
            if "Deadlock found" in str(e):
                retries += 1
                if retries < max_retries:
                    print("Deadlock encountered. Retrying...")
                    time.sleep(1)  # Add delay before retrying
                else:
                    print("Max retries reached. Giving up.")
                    raise  # Re-raise the exception if max retries reached
            else:
                # If it's not a deadlock, raise the exception immediately
                raise
    
def get_media_abs_path(customerID, filename):
    path_to_save = ".." + config['save_customer_path_parent'] + config['save_customer_path_child']
    media_directory = os.path.join(os.getcwd(), path_to_save)
    new_folder_path = os.path.normpath(os.path.join(media_directory, str(customerID)))
    os.makedirs(new_folder_path, exist_ok=True)
    full_image_file_path = os.path.join(new_folder_path, filename)
    return full_image_file_path

def get_customer_vid_DB_path(customerID, filename):
    relative_image_file_path = os.path.normpath(os.path.join(config['save_customer_path_child'], str(customerID), filename))
    return relative_image_file_path

def relate2abs_cvt(filename, related_path_to_save):
    media_directory = os.path.join(os.getcwd(), related_path_to_save)
    new_folder_path = os.path.normpath(media_directory)
    os.makedirs(new_folder_path, exist_ok=True)
    full_image_file_path = os.path.join(new_folder_path, filename)
    return full_image_file_path

def get_djangoapp_path(filename, related_path_to_save):
    relative_image_file_path = os.path.normpath(os.path.join(related_path_to_save, filename))
    return relative_image_file_path

def put_text_anywhere(frame, text_to_put_list:list, start_position_x, start_position_y):
    for text in text_to_put_list:
        cv2.putText(frame, 
                text,
                (start_position_x,start_position_y),
                font,       #font name
                1,          #font scale
                (0,0,255),  #font color
                1           #font thickness
        )
        start_position_y += 30

def classify_unknown_or_customer(people_dict, known_employee, id, is_customer, frame_count, present_datetime, center_coord, top_left, bottom_right, frame_data):

    if is_customer:
        data = people_dict[id]
        probability_is_customer = data.probToBeCustomer
        found_amount = (data.frame_latest_found - data.frame_first_found) + 1
        #edit the prob
        probability_is_customer = ((probability_is_customer * found_amount) + 1) / (found_amount + 1)
        #update latest_found_frame and update dict
        latest_found_frame = frame_count
        data.frame_latest_found = latest_found_frame
        data.probToBeCustomer = probability_is_customer
        data.dt_latest_found = present_datetime
        data.add_pixel(center_coord)
        data.top_left = top_left
        data.bottom_right = bottom_right
        people_dict[id] = data
    elif not is_customer :
        data = people_dict[id]
        probability_is_customer = data.probToBeCustomer
        found_amount = (data.frame_latest_found - data.frame_first_found) + 1
        #edit the prob
        probability_is_customer = (probability_is_customer * found_amount) / (found_amount + 1)
        #update latest_found_frame and update dict
        if probability_is_customer < 0.1:
            probability_is_customer = 0
        latest_found_frame = frame_count
        data.frame_latest_found = latest_found_frame
        data.probToBeCustomer = probability_is_customer
        data.dt_latest_found = present_datetime
        data.add_pixel(center_coord)
        data.top_left = top_left
        data.bottom_right = bottom_right
        people_dict[id] = data
    #if it's unknown or customer
    if people_dict[id].fixed == False:
        if (people_dict[id].probToBeCustomer > 0.5 and people_dict[id].person_type == "unknown"):
            data = people_dict[id]
            data.person_type = "customer"
            people_dict[id] = data
        elif (people_dict[id].probToBeCustomer < 0.5 and people_dict[id].person_type == "customer"):
            data = people_dict[id]
            data.person_type = "unknown"
            people_dict[id] = data
    


        #fix it
        if (people_dict[id].probToBeCustomer > 0.5 and 
                people_dict[id].person_type == "customer" and 
                (people_dict[id].dt_latest_found - people_dict[id].dt_first_found).total_seconds() > 5
                and people_dict[id].fixed == False):
            data = people_dict[id]
            data.person_type = "customer"
            data.fixed = True
            people_dict[id] = data
        elif id in known_employee.keys():
            data = people_dict[id]
            data.person_type = known_employee[id]
            data.fixed = True
            people_dict[id] = data
            del known_employee[id]
    #if it's employee and fixed
    elif people_dict[id].fixed == True and people_dict[id].person_type != "customer" and people_dict[id].person_type != "unknown":
        #if employee has exchange with customer then have chance to reset to be unknown
        if (people_dict[id].probToBeCustomer > 0.5 and 
                people_dict[id].person_type != "customer" and people_dict[id].person_type != "unknown" and
                people_dict[id].fixed == True):
            data = people_dict[id]
            data.person_type = "unknown"
            data.fixed = False
            data.saved_frame.clear()
            people_dict[id] = data
        #if it was employee then save the frame 
        """if (people_dict[id].person_type != "customer" and people_dict[id].person_type != "unknown" and 
            (present_datetime - people_dict[id].saved_frame_latest_dt).total_seconds() >= 4):
            print(f"IM FUCKING ADD Frame FOR ID {id}")

            data = people_dict[id]

            #delete the exceed frame limit
            if len(data.saved_frame) > 15:
                data.saved_frame.pop(0)

            print(f"len(data.saved_frame) = {len(data.saved_frame)}")
                
            data.saved_frame_latest_dt = present_datetime
            data.saved_frame.append(frame_data)
            people_dict[id] = data"""
            



def print_queue(queue):
    # Create a temporary list to store the items
    temp_list = []
    while not queue.empty():
        item = queue.get()
        temp_list.append(item)
        print("Item:", item)
    
    # Put the items back into the queue
    for item in temp_list:
        queue.put(item)


def update_local_dict(local_dict, key_contain_in_frame, now_frame):
    #keys_to_delete = [key for key in local_dict if key not in key_contain_in_frame]
    keys_to_delete = [key for key in local_dict if (now_frame - local_dict[key].frame_latest_found) > 8]
    for key in keys_to_delete:
        # If the key is not in the list of IDs, delete it from the dictionary
        del local_dict[key]

if __name__ == "__main__":
    #update_db("test", "name", "sukei", ["address = 'Highway21'", "text2 = 'suk'"])
    # Define the relative path to "djangoAPP/mock_media"
    _, sus_id = select_db('suspicious_events', ['max(sus_ID)'], [f"sus_where = 0"])
    print(sus_id[0][0])