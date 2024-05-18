import cv2
import numpy as np
import queue
import os

if __name__ == "__main__":
    from function_bundle import *
    from ultralytics import YOLO
    import multiprocessing
    from encoding_known_face import *
    import threading
    import traceback
    import statistics
    import logging
    import sys
    availability_cache = [None for _ in range (len(table_points))]
    list_total_count_cache = [0 for _ in range (len(table_points))]


frame_count = 0
total_frame_count = 0
frame_rate = 0
stop_thread = False
check_available_started = False
list_realtime_count_cache = []
count_person_at_table_cache = []
end_recording = False
check_dimsum_started = False
fakeCamFrame = None
simulate_status = None
update_shared_dict_in_procress = False
stop_check_drawer_open = False
id_at_table_cache = []

person_in_cashier_cache = None
drawer_observing = False
check_employee_too_long_started = False
check_drawer_open_started = False
cashier_aleart_set = False




####################### THREADING PROCESS {BEGIN} #######################
def calculate_real_people_total():
    global list_total_count_cache, stop_thread
    #initialize sampling list
    sampling_amount = 100
    sampling_from_tables = []
    table_name = "customer_events"

    for _ in range (len(table_points)):
        sampling_from_tables.append([])

    try:
        while not stop_thread:
            list_total_count = list_realtime_count_cache.copy()
            

            target_time = present_datetime + timedelta(seconds=4)

            # operating
            for i in range (len(table_points)):

            
                # if occupied then keep collecting until it reaches 100 collection
                if availability_cache[i] == "occupied" and len(sampling_from_tables[i]) < sampling_amount:
                    realtime_count_that_table = list_total_count[i]
                    sampling_from_tables[i].append(realtime_count_that_table)

                elif availability_cache[i] == "unoccupied": #if become unoccupied then reset
                    sampling_from_tables[i].clear()
                    
                

            # determine the meaning
            for i in range (len(table_points)):
                if len(sampling_from_tables[i]) > 0: # have sampling data
                    list_total_count[i] = statistics.mode(sampling_from_tables[i])
                    #list_total_count[i] = max(sampling_from_tables[i])
                    #list_total_count[i] = round(sum(sampling_from_tables[i])/len(sampling_from_tables[i]))
                    if (len(sampling_from_tables[i]) == 1 or 
                        len(sampling_from_tables[i]) % 25 == 0 or 
                        len(sampling_from_tables[i])) == 10:
                        #UPDATE customer_events SET customer_amount = %s WHERE 
                        #customer_IN = (SELECT MAX(customer_IN) from customer_events WHERE tableID = %s)
                        update_db(table_name, "customer_amount", list_total_count[i], 
                                ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})",
                                f"tableID = {i+1}"])
                        
                else: # have no sampling data
                    list_total_count[i] = 0
            list_total_count_cache = list_total_count.copy()
            list_total_count.clear()

            while present_datetime < target_time and not stop_thread:
                time.sleep(1)

    except Exception as e:
        print("error: ", e)
        traceback.print_exc()
        stop_thread = True

    #save befor exit
    for i in range (len(table_points)):
        if len(sampling_from_tables[i]) > 0:
            real_amount = int(statistics.mode(sampling_from_tables[i]))
            update_db(table_name, "customer_amount", real_amount, 
                ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})",
                f"tableID = {i+1}"])
            print("Updated!!!")
    print('\033[93m' + 'calculate_real_people_total stopped' + '\033[0m')


def check_available():
    global list_realtime_count_cache, stop_thread, availability_cache, check_available_started, fps, frame_rate, list_time_start

    table_name = "customer_events"
    itr = 0

    while not stop_thread:
        itr += 1
        itr1 = 0
        table_status = [] # 3 = occupied , 0 = unoccupied
        availability = []

        try:
            for table_no, _ in enumerate(table_points): #initialize list
                table_status.append(0)
            # start checking for 3 sampling time at each tables
            for i in range (0, 5):
                target_time = present_datetime + timedelta(seconds=4)
                itr1 += 1
                list_realtime_count = list_realtime_count_cache.copy()
                for table_no, _ in enumerate(table_points):
                    if list_realtime_count[table_no] > 0: #occupied detected that moment
                        table_status[table_no] += 1
                    
                    if stop_thread:
                        break
                while present_datetime < target_time and not stop_thread:
                    time.sleep(1)

                if stop_thread:
                    break
            if stop_thread:
                break

            # determining the meaning of state
            for i, item in enumerate(table_status):
                if item >= 5/2:   # occupied
                    availability.append("occupied")
                    if availability_cache[i] == "unoccupied" or  availability_cache[i] == None:
                        start_occupied_datetime[i] = present_datetime
                        field_list = ["tableID", "customer_IN", "created_datetime"]
                        value_list = [i+1, present_datetime, datetime.now()]
                        insert_db(table_name, field_list, value_list)
                else:                   #unoccupied
                    availability.append("unoccupied")
                    if availability_cache[i] == "occupied"  or  availability_cache[i] == None:
                        start_occupied_datetime[i] = None
                        update_db(table_name, "customer_OUT", present_datetime, 
                                  ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})",
                                   f"tableID = {i+1}"])

            availability_cache = availability.copy()
            if check_available_started == False:
                check_available_started = True

        except Exception as e:
            print("error: ", e)
            traceback.print_exc()
            stop_thread = True
    #save before exit
    for i, item in enumerate(table_status):
        if table_status[i] > 5/2:
            update_db(table_name, "customer_OUT", present_datetime, 
                ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})",
                f"tableID = {i+1}", "customer_OUT = '0000-00-00 00:00:00'"])
    print('\033[93m' + 'check_available stopped' + '\033[0m')

def now_frame_rate():
    global frame_rate, stop_thread
    second = 1
    while not stop_thread:
        frame_count_before = frame_count
        time.sleep(second)
        frame_count_after = frame_count
        frame_rate = int((frame_count_after - frame_count_before)/second)
    print('\033[93m' + 'now_frame_rate stopped' + '\033[0m')

def combine_frame():
    print('\033[91m' + '** start saving preview as video' + '\033[0m')
    completed_frames = None
    global stop_thread
    # Define video resolution and frame rate
    frame_width = 1920
    frame_height = 1080

    size = (frame_width, frame_height) 
    # Define codec and create VideoWriter object
    out = cv2.VideoWriter('result_video/processing_record.mp4', 
                            cv2.VideoWriter_fourcc(*'H264'), 
                            30, size) 

    while not stop_thread and not end_recording:
        # Iterate over each video queue to check for frames
        if completed_frames is None:  # Check if frame for this index is not yet filled
            frame = object1.get_frame()  # Get frame from video queue
            
            if frame is not None:  # If frame is available, update completed_frames
                completed_frames = frame
                
        if frame is not None:
            completed_frames = cv2.resize(completed_frames, (1920,1080))
            out.write(completed_frames)
            # Display the combined frame
            cv2.imshow('completed_frames', completed_frames)
            completed_frames = None
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.1)
    out.release()
    print('\033[93m' + 'combine_frame stopped' + '\033[0m')

def check_dimsum(table_index, object_frame_in):
    print('\033[91m' + f'*check_dimsum at table {table_index+1} started' + '\033[0m')
    model = YOLO(config['dimsum_model_path'])
    global stop_thread, realtime_dimsum_found, check_dimsum_started
    to_check[table_index] = 2
    frame_point = table_crop_points[table_index]
    x_min = frame_point[0][0]
    y_min = frame_point[0][1]
    x_max = frame_point[2][0]
    y_max = frame_point[2][1]
    found = 0
    itr = 0

    while not stop_thread and not stop_dimsum_thread[table_index]:
        if not object_frame_in.is_empty():
            itr += 1
            frame_obj = object_frame_in.get_frame_obj()
            frame = frame_obj.frame
            frame_datetime = frame_obj.date_time
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            results = model(source=[cropped_frame], conf=0.6, show=False, save=False, classes=[0], verbose=False)
            realtime_dimsum_found[table_index] = len(results[0])
            annotated_frame = results[0].plot()
            if len(results[0]) > 0:
                found += 1

        if itr == 10 and found >= 6:
            to_check[table_index] = 3
            update_db("customer_events", "time_getFood", frame_datetime, 
                      ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {table_index+1}'])[0]})", 
                       f"tableID = {table_index+1}"])
            
            add_jpg_media(table_index, "getFoodFrame.jpg", annotated_frame)

            break
        elif itr == 10 and found < 6:
            itr = 0
            found = 0
        time.sleep(0.1)
    object_frame_in.clear_all()
    print('\033[93m' + f'check_dimsum at table {table_index+1} stopped' + '\033[0m')

def FakeCamera():
    print('\033[93m' + 'FakeCamera started' + '\033[0m')

    global fps, simulate_status, stop_thread
    """Reads the video file at its natural rate, storing the frame in a global called 'fakeCamFrame'"""
    logging.debug(f'[FakeCamera] Generating video stream from {url_path}')

    # Open video
    video = cap
    if (video.isOpened()== False):
       logging.critical(f'[FakeCamera] Unable to open video {url_path}')
       simulate_status = False
       return

    # Get height, width and framerate so we know how often to read a frame
    h   = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    logging.debug(f'[FakeCamera] h={h}, w={w}, fps={fps}')

    # Initialise fakeCamFrame
    global fakeCamFrame
    fakeCamFrame = np.zeros((h,w,3), dtype=np.uint8)

    # Signal main that we are ready
    simulate_status = True

    while not stop_thread:
        ret, frame = video.read()
        if ret == False:
            break
        # Store video frame where main can access it
        fakeCamFrame[:] = frame[:]
        # Try and read at appropriate rate
        time.sleep(1.0/fps)

    logging.debug('[FakeCamera] Ending')
    simulate_status = False
    print('\033[93m' + 'FakeCamera stopped' + '\033[0m')

def record_customer_activities(): #must start after check_available started only
    
    global stop_thread, present_datetime, plotted_points_recording, end_recording, availability_cache, blank_frame_cache
    record_status = [0 for _ in range (len(table_points))]
    record_object = [None for _ in range (len(table_points))]
    abs_path = [None for _ in range (len(table_points))]
    relate_path = [None for _ in range (len(table_points))]
    try:
        while not stop_thread and not end_recording:
            #before doing anything specify sleep time and copy frame first
            target_time = present_datetime + timedelta(seconds=4)

            for i in range (len(table_points)):
                if availability_cache[i] == "occupied":
                    if record_status[i] == 0:
                        print('\033[91m' + f'*record_activities at table {i+1} started' + '\033[0m')
                        #select id from customer_event where created_datetime = (max(created_datetim) from table=i+1) and table = i+1
                        _, customerID = select_db("customer_events", ["customer_ID"], 
                                                ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})", f"tableID = {i+1}"])
                        image_full_file_path = get_media_abs_path(customerID[0][0], "wrapped_up.mp4")
                        abs_path[i] = image_full_file_path      #save absolute path dir to video to list
                        image_relate_file_path = get_customer_vid_DB_path(customerID[0][0], "wrapped_up.mp4")
                        relate_path[i] = image_relate_file_path
                        
                        record_status[i] = 1

                        #start recording & initialize the video header
                        result = cv2.VideoWriter(image_full_file_path, 
                            cv2.VideoWriter_fourcc(*'H264'), 
                            1, record_width_height[i]) 
                        record_object[i] = result

                        #save filepath to database #update relative path dir of video to db
                        update_db("customer_events", "captured_video", image_relate_file_path, 
                        ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})", 
                        f"tableID = {i+1}"])

                    if record_status[i] == 1:

                        #crop frame of each table
                        y_min, y_max = plotted_points_recording[i][0][1], plotted_points_recording[i][2][1]
                        x_min, x_max = plotted_points_recording[i][0][0], plotted_points_recording[i][2][0]
                        frame = blank_frame_cache.copy()
                        cropped_frame = frame[y_min:y_max, x_min:x_max]

                        #put text of time in frame
                        cv2.putText(cropped_frame,str(present_datetime.strftime('%Y-%m-%d %H:%M:%S')),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                        #write frame
                        record_object[i].write(cropped_frame) 

                elif availability_cache[i] == "unoccupied":
                    if record_status[i] == 1:
                        print('\033[91m' + f'**record_activities at table {i+1} stopped' + '\033[0m')

                        #end recording
                        record_object[i].release()

                        #clear all status and path
                        record_status[i] = 0
                        record_object[i] = None
                        record_object[i] = None
                        abs_path[i] = None
                        relate_path[i] = None


            # sleep for specific period time.sleep end_recording must not be true
            while present_datetime < target_time and not stop_thread and not end_recording:
                time.sleep(1)
        
        #loop every record object and use obj.out() to kill the recording in case of thread killing
        for i in range (len(table_points)):
            if record_object[i] != None:
                record_object[i].release()
        print('\033[93m' + 'record_customer_activities stopped' + '\033[0m')
    except Exception as e:
        print("error: ", e)
        traceback.print_exc()
        stop_thread = True
        
def update_shared_dict():
    global stop_thread, update_shared_dict_in_procress, blank_frame_cache
    while not stop_thread:
        while shared_dict_update_queue.empty() and not stop_thread:
            time.sleep(0.1)
        if not stop_thread:
            local_dict = shared_dict_update_queue.get()
            # Create a set of keys to delete from the shared dictionary
            keys_to_delete = set(shared_dict.keys()) - set(local_dict.keys())
            shared_dict_update_inprogress.set()
            # Delete keys from the shared dictionary
            if frame_count % 5 == 0:
                face_recog_queue.put((blank_frame_cache, local_dict))
            for key in keys_to_delete:
                if stop_thread:
                    break
                del shared_dict[key]

            # Update or add keys to the shared dictionary
            shared_dict.update(local_dict)
            #print(f"shared dict after updated {shared_dict}")
            
            shared_dict_update_inprogress.clear()
    print('\033[93m' + 'update_shared_dict stopped' + '\033[0m')

def recognize_employee_face(shared_dict, todo_queue, known_face_encodings, known_face_names, stop_subprocess, known_employee, shared_dict_update_inprogress):
    
    import face_recognition
    import time
    i = 0

    try:
        while not stop_subprocess.is_set():
            filtered_dict = {}
            while todo_queue.empty() and not stop_subprocess.is_set():
                time.sleep(0.1)
            if not stop_subprocess.is_set():
                #retrieve data 
                todo = todo_queue.get()
                frame = todo[0]
                dict_from_queue = todo[1]

                #loop through the dict to find which one to recog
                for key, value in dict_from_queue.items():
                    if value.person_type == "unknown":
                        filtered_dict[key] = value


                #face recog part >>>
                # Pop items until the dictionary is empty
                while filtered_dict:
                    key, value = filtered_dict.popitem()
                    while shared_dict_update_inprogress.is_set():
                        time.sleep(0.1)
                    if value.fixed == True or key not in shared_dict:
                        continue
                    if value.fixed == False and key in shared_dict:
                        top_left = value.top_left
                        bottom_right = value.bottom_right
                        y_min = top_left[1]
                        y_max = bottom_right[1]
                        x_min = top_left[0]
                        x_max = bottom_right[0]
                        cropped_person = frame[y_min:y_max, x_min:x_max]
                        cropped_person_BGR = cropped_person
                        cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)

                        #cv2.imwrite("person_found.jpg", cropped_person_BGR)
                        #find face location and encoding
                        face_location = face_recognition.face_locations(cropped_person, model='hog')

                        face_encodings = face_recognition.face_encodings(cropped_person, face_location)

                        #face comparing
                        for itr, face_encoding in enumerate(face_encodings):

                            """i += 1
                            media_directory = os.path.join(os.getcwd(), "employee_face_result")
                            new_folder_path = os.path.normpath(media_directory)
                            os.makedirs(new_folder_path, exist_ok=True)
                            filename = f"{i}.jpg"
                            path_to_file = os.path.join(new_folder_path, filename)
                            print(path_to_file)
                            cv2.imwrite(path_to_file, cropped_person_BGR)"""

                            # Compare face encoding with known faces
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.4)
                            name = "Unknown"

                            # If a match is found, use the known face name
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = known_face_names[first_match_index]

                                #update shared dict part >>>
                                if key in shared_dict:
                                    
                                    # Update the value of the key
                                    print('\033[91m' + 'I recognized someone!!!' + '\033[0m')
                                    print(f"found {name} at id {key}")
                                    known_employee[key] = name
                                    break
            keys_to_delete = set(known_employee.keys()) - set(shared_dict.keys())
            for key in keys_to_delete:
                del known_employee[key]

    except Exception as e:
        print("error: ", e)
        traceback.print_exc()
        stop_subprocess.set()
    print('\033[93m' + f'[Subprocess] face recognition stopped' + '\033[0m')

def check_drawer_open():
    print('\033[93m' + f'check_drawer_open [started]' + '\033[0m')
    global stop_thread, check_drawer_open_started, cashier_aleart_set
    model = YOLO(config['drawer_model_path'])
    status_record = False
    open_found = 0
    itr = 25
    status_before_is_open = False
    status_now_is_open = False
    cashier_record = None
    check_drawer_open_started = True

    #coordination for detecting
    y_min, y_max = drawer_detect_points[0][0][1], drawer_detect_points[0][2][1]
    x_min, x_max = drawer_detect_points[0][0][0], drawer_detect_points[0][2][0]

    y_min_rec, y_max_rec = cashier_area_record[0][0][1], cashier_area_record[0][2][1]
    x_min_rec, x_max_rec = cashier_area_record[0][0][0], cashier_area_record[0][2][0]
    size = cashier_record_width_height[0]

    employeeID = None

    try:
        #start loop
        
        while not stop_thread and not stop_check_drawer_open:
            
                    
            #detect if drawer is at least half of 5 frame 
            for _ in range (itr):
                cashier_area = blank_frame_cache[y_min:y_max, x_min:x_max]
                cashier_area_rec = blank_frame_cache[y_min_rec:y_max_rec, x_min_rec:x_max_rec]
                

                #use model detection
                
                results = model(source=[cashier_area], conf=0.6, show=False, save=False, classes=[0], verbose=False)
                annotated_frame = results[0].plot()

                result_length = results[0].cpu().numpy()
                if len(result_length) != 0:
                
                    for i in range(len(results[0])):

                        boxes = results[0].boxes
                        box = boxes[i]  # returns one box
                        bb = box.xyxy.cpu().numpy()[0]
                        x_rec_min = int(bb[0]+x_min-x_min_rec)
                        y_rec_min = int(bb[1]+y_min-y_min_rec)
                        x_rec_max = int(bb[2]+x_min-x_min_rec)
                        y_rec_max = int(bb[3]+y_min-y_min_rec)


                        cv2.rectangle(
                            cashier_area_rec,
                            (x_rec_min, y_rec_min),
                            (x_rec_max, y_rec_max),
                            (0, 0, 255),
                            3,
                        )

                # Convert tensor array to numpy
                if len(results[0]) > 0:
                    open_found += 1
                cv2.imshow("cashier", annotated_frame)
                cv2.waitKey(1)

            #Interpret the status and reset open_found
            if open_found > itr/2:      #if found opened more than half of 5 frame = Opened
                status_before_is_open = status_now_is_open
                status_now_is_open = True
            else:
                status_before_is_open = status_now_is_open
                status_now_is_open = False
            open_found = 0
            
            person_in_cashier_now = person_in_cashier_cache #id
            #Case 1: if drawer were detected open
            if status_before_is_open == False and status_now_is_open == True:

                cashier_aleart_set = True

                #fetch data from person at cashier
                while shared_dict_update_inprogress.is_set():
                    time.sleep(0.1)
                if person_in_cashier_now not in shared_dict:
                    continue

                person_data = shared_dict[person_in_cashier_now]
                
                #set name path with datetimenow
                video_filename = present_datetime.strftime("%Y%m%d%H%M%S" + ".mp4")
                path_to_save_PC = "../djangoAPP/mock_media/drawer_sus"
                path_to_save_DB = "/mock_media/drawer_sus"
                absolute_path = relate2abs_cvt(video_filename, path_to_save_PC)
                db_path = get_djangoapp_path(video_filename, path_to_save_DB)
                
                #initializing video writer
                cashier_record = cv2.VideoWriter(absolute_path, cv2.VideoWriter_fourcc(*'H264'), 7, size) 
                
                #insert suspicious in DB
                field_list = ["sus_type", "sus_employeeID", "sus_video", "sus_status", "sus_datetime", "sus_where"]

                #if know employee name at first glance 
                if (person_data.person_type != "unknown" and person_data.person_type != "customer" and employeeID == None):
                    _, employeeID = select_db("employee", ["employee_ID"], [f"employee_name = '{person_data.person_type}'"], verbose = True)
                    value_list = [0, employeeID[0][0], db_path, 1, present_datetime, 0]

                    #prevent dict error before insert suspicious event
                    if person_in_cashier_now in shared_dict:
                        insert_db("suspicious_events", field_list, value_list, verbose=True)

                        #put text of time in frame
                        cv2.putText(cashier_area_rec,str(present_datetime.strftime('%Y-%m-%d %H:%M:%S')),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                        cashier_record.write(cashier_area_rec)

                        status_record = not status_record
                        _, sus_id = select_db('suspicious_events', ['max(sus_ID)'], [f"sus_where = 0"])
                        print('\033[91m' + f'**check_drawer_open started recording with knowing who open' + '\033[0m')

                #if don't know employee name at first glance
                elif (person_data.person_type == "unknown" and employeeID == None):
                    value_list = [0, None, db_path, 1, present_datetime, 0]

                    #prevent dict error before insert suspicious event
                    if person_in_cashier_now in shared_dict:
                        insert_db("suspicious_events", field_list, value_list)
                        cv2.putText(cashier_area_rec,str(present_datetime.strftime('%Y-%m-%d %H:%M:%S')),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                        cashier_record.write(cashier_area_rec)
                        status_record = not status_record
                        _, sus_id = select_db('suspicious_events', ['max(sus_ID)'], [f"sus_where = 0"])
                        print('\033[91m' + f'**check_drawer_open started recording WITHOUT knowing who open' + '\033[0m')

            #Case 2: if the drawer is already opened
            elif status_before_is_open == True and status_now_is_open == True:
                cv2.putText(cashier_area_rec,str(present_datetime.strftime('%Y-%m-%d %H:%M:%S')),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                cashier_record.write(cashier_area_rec)

                #prevent dict error
                while shared_dict_update_inprogress.is_set():
                    time.sleep(0.1)
                if person_in_cashier_now not in shared_dict:
                    continue
                person_data = shared_dict[person_in_cashier_now]

                #if program has figured out what employee was or person still unknow, then just let go
                if person_data.person_type == "unknown" or employeeID != None: #if figured out what employee was, then just let go
                    pass
                
                #if program just already know name of person and never knew the name of person before, then
                #update the name to suspicious record
                elif person_data.person_type != "unknown" and employeeID == None:
                    person_name = person_data.person_type
                    _, employeeID = select_db("employee", ["employee_ID"], [f"employee_name = '{person_name}'"], verbose = True)
                    #>>>>>>>update database for employee name
                    update_db("suspicious_events", "sus_employeeID", employeeID[0][0], [f"sus_ID = {sus_id[0][0]}"], verbose = True)
                    print('\033[93m' + f'**check_drawer_open know who open the drawer: {person_name}' + '\033[0m')

            #if the drawer is detected closed
            elif status_before_is_open == True and status_now_is_open == False:
                cashier_aleart_set = False
                status_record = not status_record
                employeeID = None
                cashier_record.release()
                print('\033[91m' + f'**!!check_drawer_open stopped recording' + '\033[0m')
    except Exception as e:
            print("error: ", e)
            traceback.print_exc()
            stop_thread = True
            stop_subprocess.set()
    if cashier_record is not None and cashier_record.isOpened():
        cashier_record.release()
    check_drawer_open_started = False
    print('\033[93m' + f'check_drawer_open stopped' + '\033[0m')

def check_employee_too_long():
    print('\033[93m' + f'check_employee_too_long started' + '\033[0m')
    
    global count_person_at_table_cache, stop_thread, availability_cache, check_available_started, fps, frame_rate, check_employee_too_long_started

    check_employee_too_long_started = True
    itr = 0
    frame_cache = []
    record_object = [None for _ in range (len(employee_detect_area_points))]
    fuck = 0
    sus_id_list = [None for _ in range (len(employee_detect_area_points))]
    person_type_cache = [None for _ in range (len(employee_detect_area_points))]

    while not stop_thread:
        itr += 1
        itr1 = 0
        employee_at_table_status = [] # 3 = occupied , 0 = unoccupied
        employee_occupied = [None for _ in range (len(employee_detect_area_points))]
        length = []
        frame_cache = [[] for _ in range (len(employee_detect_area_points))]
        timeframe_cache = [[] for _ in range (len(employee_detect_area_points))]

        try:
            for table_no, _ in enumerate(employee_detect_area_points): #initialize list
                employee_at_table_status.append(0)
            # start checking for 3 sampling time at each tables
            for i in range (0, 15):
                target_time = present_datetime + timedelta(seconds = 2)
                itr1 += 1
                count_person_at_table = count_person_at_table_cache.copy()
                for table_no, _ in enumerate(employee_detect_area_points):
                    if availability_cache[table_no] == "occupied":

                        y_min_rec, y_max_rec = employee_record_area_point[table_no][0][1], employee_record_area_point[table_no][2][1]
                        x_min_rec, x_max_rec = employee_record_area_point[table_no][0][0], employee_record_area_point[table_no][2][0]

                        if len(frame_cache[table_no]) > 20:
                            frame_cache[table_no].pop(0)

                        frame = blank_frame_cache[y_min_rec:y_max_rec, x_min_rec:x_max_rec]
                        
                        frame_cache[table_no].append(frame)
                        timeframe_cache[table_no].append(present_datetime)

                        
                        if count_person_at_table[table_no] > 0: #occupied detected that moment
                            employee_at_table_status[table_no] += 1
                        
                        if stop_thread:
                            break
                while present_datetime < target_time and not stop_thread:
                    time.sleep(1)

                if stop_thread:
                    break
            
            for item in frame_cache:
                length.append(len(item))
            length.clear()
            if stop_thread:
                break

            # determining the meaning of state
            for table_no, status_count in enumerate(employee_at_table_status):
                if availability_cache[table_no] == "occupied":
                    if status_count >= 15/2:   # occupied
                        employee_occupied[table_no] = "occupied"
                        if record_object[table_no] is None:

                            #find out the employeeID
                            if person_type_cache[table_no] == None:
                                id_at_table = id_at_table_cache[table_no]
                                person_type = shared_dict[id_at_table].person_type
                                person_type_cache[table_no] = person_type

                            #set name path with datetimenow
                            video_filename = present_datetime.strftime("%Y%m%d%H%M%S" + ".mp4")
                            path_to_save_PC = "../djangoAPP/mock_media/drawer_sus"
                            path_to_save_DB = "/mock_media/drawer_sus"
                            absolute_path = relate2abs_cvt(video_filename, path_to_save_PC)
                            db_path = get_djangoapp_path(video_filename, path_to_save_DB)

                            #start initializing recording
                            size = employee_record_width_height[table_no]
                            objrec = cv2.VideoWriter(absolute_path, cv2.VideoWriter_fourcc(*'H264'), 7, size)
                            record_object[table_no] = objrec        
                            while len(frame_cache[table_no]) > 0:
                                fuck += 1
                                frame = frame_cache[table_no].pop(0)
                                timeframe = timeframe_cache[table_no].pop(0)
                                #cv2.imwrite(f"fukkkkif{fuck}.jpg", frame)
                                cv2.putText(frame,str(timeframe.strftime('%Y-%m-%d %H:%M:%S')),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                                record_object[table_no].write(frame)

                            
                            _, employeeID = select_db("employee", ["employee_ID"], [f"employee_name = '{person_type_cache[table_no]}'"], verbose = True)
                            
                            #insert into database
                            field_list = ["sus_type", "sus_employeeID", "sus_video", "sus_status", "sus_datetime", "sus_where"]
                            value_list = [1, employeeID[0][0], db_path, 1, present_datetime, table_no+1]
                            insert_db("suspicious_events", field_list, value_list, verbose=True)
                            _, sus_id = select_db('suspicious_events', ['max(sus_ID)'], [f"sus_type = 1"])
                            sus_id_list[table_no] = sus_id[0][0]
                            

                        elif record_object[table_no] is not None:
                            #continue feed frame for recording
                            while len(frame_cache[table_no]) > 0:
                                fuck += 1
                                frame = frame_cache[table_no].pop(0)
                                timeframe = timeframe_cache[table_no].pop(0)
                                #cv2.imwrite(f"fukkkkelif{fuck}.jpg", frame)
                                cv2.putText(frame,str(timeframe.strftime('%Y-%m-%d %H:%M:%S')),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                                record_object[table_no].write(frame)
                        
                    else:                   #unoccupied
                        employee_occupied[table_no] = "unoccupied"
                        if record_object[table_no] is None:
                            #Do nothing
                            pass
                        elif record_object[table_no] is not None:
                            #release recording
                            sus_id_list[table_no] = None
                            record_object[table_no].release()
                            record_object[table_no] = None
                            person_type_cache[table_no] = None

            employee_occupied_cache = employee_occupied.copy()

        except Exception as e:
            print("error: ", e)
            traceback.print_exc()
            stop_thread = True
            stop_subprocess.set()
    for item in record_object:
        if item is not None and item.isOpened():
            item.release()
    print('\033[93m' + f'check_employee_too_long stopped' + '\033[0m')



def print_shared_key():
    while not stop_thread:
        for key, value in known_employee.items():
            print(key, ":", value, sep=" ", end="")
            print("   ",end="")
        time.sleep(0.1)
        print("\n")
    print('\033[93m' + f'printed_shared_key stopped' + '\033[0m')
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

####################### THREADING PROCESS {END} #######################
####################### THREADING PROCESS {END} #######################


def main(source_platform, simulate, source_url, frame_skip, date_time):
    ## delete incomplete data with no customer out time before the process

    global stop_thread, frame_count, fps, frame_rate, present_datetime, list_realtime_count_cache, object2, fakeCamFrame, simulate_status, end_recording
    global blank_frame_cache, total_frame_count, stop_check_drawer_open, person_in_cashier_cache, drawer_observing, count_person_at_table_cache, id_at_table_cache
    found_in_cashier_count = 0
    drawer_thread  = None
    person_in_cashier = None
    brightness_increase = 50
    #delete incomplete data before start program
    i = "0000-00-00 00:00:00"
    condition_list = [f"customer_OUT = '{i}'"]
    try:
        delete_data_db("customer_events", condition_list)
    except Exception as e:
        print('\033[91m' + f"error: {e}" + '\033[0m')
        stop_thread = True
        stop_subprocess.set()
        exit()

    ### Start reading frame ###
    if not simulate and  not cap.isOpened():
        print("Cannot open camera")
        exit()
    elif simulate and source_platform == "video_frame":

        # Create a fake camera thread that reads the video in "real-time"
        if total_frame_count == 0:
            fakeCamThread = threading.Thread(target=FakeCamera)
            fakeCamThread.start()

        # Wait for fake camera to intialise
        logging.debug(f'[main] Waiting for camera to power up and initialise')
        while simulate_status != True:
            time.sleep(0.1)
        ret = simulate_status
        if ret != True:
            print("Cannot open camera")
            stop_thread = True
            sys.exit()

    thread1 = threading.Thread(target=calculate_real_people_total)
    thread2 = threading.Thread(target=check_available)
    thread3 = threading.Thread(target=now_frame_rate)
    thread4 = threading.Thread(target=combine_frame)
    thread5 = threading.Thread(target=record_customer_activities)
    thread6 = threading.Thread(target=update_shared_dict)
    thread7 = threading.Thread(target=print_shared_key)
    thread8 = threading.Thread(target=check_employee_too_long)
    
    print('\033[92m' + f'Initializing Done!!!' + '\033[0m')
    print('\033[93m' + f'[Main Loop] Started!' + '\033[0m')
    while not stop_thread:
        
        
        # Capture frame-by-frame
        if not simulate and source_platform == "video_frame":    #video frame and not simulate fake camera
            ret, frame = cap.read() 
            present_datetime = present_datetime + timedelta(seconds=1/fps)
        elif simulate and source_platform == "video_frame":      #video frame and simulate fake camera
            present_datetime = datetime.now()
            ret, frame = simulate_status, fakeCamFrame.copy()
            """frame = cv2.putText(frame, 
                    "frame " + str(total_frame_count),
                    (500, 650),
                    cv2.FONT_HERSHEY_COMPLEX,       #font name
                    1,          #font scale
                    (0,0,255),  #font color
                    2           #font thickness
            )
            # Show the current frame
            cv2.imshow('fakevid', frame)"""
        elif not simulate and source_platform == "live_frame":   #live frame
            ret, frame = cap.read() 
            frame = cv2.resize(frame, (1920, 1080))

            present_datetime = datetime.now()

        

        # Add the brightness increase value to each pixel, clip the values to stay within [0, 255]
        #frame = adjust_brightness_contrast(frame, alpha=1.2, beta=50)
        #frame = sharpen_image(frame)
        #frame = cv2.add(frame, np.ones(frame.shape, dtype=np.uint8) * brightness_increase)
        frame_obj = frame_attr(frame, present_datetime)
        frame_data = frame_obj.frame
        frame_time = frame_obj.date_time
        
        # Frame counting
        total_frame_count += 1

        id_at_table = [None for _ in employee_record_area_point]

        ### Process the frame skipped ###
        if total_frame_count == 1 or (total_frame_count) % frame_skip == 0:
            if not ret or stop_subprocess.is_set():
                print('\033[92m' + "Can't receive frame (stream end?). Exiting ..." + '\033[0m')
                stop_thread = True
                end_recording = True
                stop_subprocess.set()
                break
            frame_count += 1
            blank_frame = frame_data.copy()
            blank_frame_obj = frame_attr(blank_frame, frame_time)
            blank_frame_cache = blank_frame

            ### Predict on image ###
            detect_params = model.track(source=[frame_data], conf=0.4, show=False, save=False, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)

            # Convert tensor array to numpy
            DP = detect_params[0].cpu().numpy()
            #print(DP.boxes)
            #print("len", len(detect_params[0]))
            key_contain_in_frame = []
            

            if len(DP) != 0:
                
                while shared_dict_update_inprogress.is_set():
                    time.sleep(0.1)
                local_dict = dict(shared_dict)
                
                for i in range(len(detect_params[0])):

                    boxes = detect_params[0].boxes
                    box = boxes[i]  # returns one box
                    clsID = box.cls.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    bb = box.xyxy.cpu().numpy()[0]

                    font = cv2.FONT_HERSHEY_COMPLEX
                    
                    # show center of a person on the frame
                    horizon_center = int((int(bb[2]) + int(bb[0]))/2)
                    vertical_center = int((int(bb[3]) + int(bb[1]))/2)
                    center_coordinate = (horizon_center, vertical_center)
                    
                    #cv2.circle(frame_data, (horizon_center, vertical_center), 10, (0,0,255), -1)

                    id = int(boxes[i].id.cpu().numpy()[0])

                    #if human dict don't have this id
                    if not id in local_dict:
                        data = person("unknown", frame_count, frame_count, 0, present_datetime, 
                                      present_datetime, False, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])))
                        data.add_pixel(center_coordinate)
                        local_dict[id] = data

                    is_customer = count_table_people(horizon_center, vertical_center)
                    
                    classify_unknown_or_customer(local_dict, known_employee, id, is_customer, frame_count, present_datetime, 
                                              (horizon_center, vertical_center),(int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                                              frame_data)
                    
                    if local_dict[id].person_type != "unknown" and local_dict[id].person_type != "customer":
                        success, is_at_table = check_employee_at_table(horizon_center, vertical_center)
                        #print("is_at_table = ", is_at_table)
                        if success:
                            id_at_table[is_at_table] = id


                    key_contain_in_frame.append(id)

                    #check if at cashier
                    is_at_cashier = check_person_at_cashier(horizon_center, vertical_center)
                    if is_at_cashier:                  
                        person_in_cashier = id
                        found_in_cashier_count+= 1
                        
                    # Display plotting class name and confidence (only used in track mode)
                    try:
                        id = int(boxes[i].id.cpu().numpy()[0])
                        
                        for pt1, pt2 in local_dict[id].all_edge():
                            cv2.line(frame_data, pt1, pt2, 
                                     person_color[int(0 if local_dict[id].person_type == "unknown" else (1 if local_dict[id].person_type == "customer" else 2))], thickness=2, lineType=cv2.LINE_AA)
                        cv2.rectangle(
                            frame_data,
                            (int(bb[0]), int(bb[1])),
                            (int(bb[2]), int(bb[3])),
                            person_color[int(0 if local_dict[id].person_type == "unknown" else (1 if local_dict[id].person_type == "customer" else 2))],
                            2,
                        )
                        cv2.rectangle(frame_data, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[1]+30)), 
                                      person_color[int(0 if local_dict[id].person_type == "unknown" else (1 if local_dict[id].person_type == "customer" else 2))], -1) 
                        cv2.putText(
                            frame_data,
                            str(local_dict[id].person_type) + " " + str(id),
                            (int(bb[0]), int(bb[1]) + 25),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )
                        cv2.putText(
                            frame_data,
                            str(local_dict[id].fixed) + "{:.2f}".format(local_dict[id].probToBeCustomer),
                            (int(bb[0]), int(bb[1]) + 50),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )
                        
                    except Exception as e:
                        print("error: ", e)
                        traceback.print_exc()
                        stop_thread = True
            #remove key that don't have now
            if len(key_contain_in_frame) > 0:
                update_local_dict(local_dict, key_contain_in_frame, frame_count)
                shared_dict_update_queue.put(local_dict)
            
            person_in_cashier_cache = person_in_cashier

            #drawer status 0 = nothing to do, 1 to do/doing 
            if frame_count % 15 == 0:
                if found_in_cashier_count > 15/3:
                    if drawer_observing == False:
                        stop_check_drawer_open = False
                        drawer_thread = threading.Thread(target=check_drawer_open)
                        drawer_thread.start()
                        drawer_observing = True
                else:
                    if drawer_observing == True:
                        stop_check_drawer_open = True
                        drawer_thread.join()
                        drawer_observing = False
                        person_in_cashier = None
                found_in_cashier_count = 0
            

            ### draw table area ###
            draw_table_point_no_border(frame_data, availability_cache, present_datetime, list_total_count_cache)
            if check_drawer_open_started and not cashier_aleart_set:
                draw_from_points(frame_data, cashier_area_points, (0, 255, 255))
            elif check_drawer_open_started and cashier_aleart_set:
                draw_from_points(frame_data, cashier_area_points, (0, 0, 255))
            """draw_from_points(frame_data, table_crop_points, (255, 0, 0))
            draw_from_points(frame_data, employee_record_area_point, (255, 0, 255))
            #draw_from_points(frame_data, plotted_points_recording, (0, 255, 255))
            draw_from_points(frame_data, drawer_detect_points, (0, 255, 255))
            draw_from_points(frame_data, cashier_area_record, (0, 255, 0))
            draw_from_points(frame_data, employee_detect_area_points, (255, 255, 0))"""
            

            
            # put text on the bottom right bottom
            text_to_put_list = []
            text_to_put_list.append("frame " + str(frame_count) + " | " + str(frame_rate) + " Frame/s | " + "totalframe " + str(total_frame_count))
            text_to_put_list.append(str(len(detect_params[0])) + " " + "person")
            text_to_put_list.append("realtime: " + str(list_realtime_count))
            text_to_put_list.append("total: " + str(list_total_count_cache))
            text_to_put_list.append(str(availability_cache))
            text_to_put_list.append("to check" + str(to_check))
            text_to_put_list.append("dimsum found" + str(realtime_dimsum_found))
            text_to_put_list.append("time now: " + str(frame_obj.date_time))
            text_to_put_list.append("fps: " + str(fps))
            put_text_bottom_right(frame_data, text_to_put_list)


            # Display the resulting frame
            cv2.imshow("ObjectDetection", frame_data)
            #cv2.imwrite("sukrit_restaurant.jpg", frame) 
            list_realtime_count_cache = list_realtime_count.copy()
            count_person_at_table_cache = count_person_at_table.copy()
            id_at_table_cache = id_at_table.copy()
            reset_people_count()
            reset_employee_at_table_count()
            
            if save_preview:
                object1.add_frame(frame_data) # uncomment without recording cause memory leak!

            if frame_count == 1:
                thread2.start() #check availability
                thread3.start()#check framerate
                if save_preview:
                    thread4.start() #record video
                thread6.start()
            #if frame_count == 20:
                #thread7.start()
            # Terminate run when "Q" pressed
            if check_available_started and not thread1.is_alive():
                thread1.start() #calculate total person
                thread5.start()
                thread8.start()

            
            if check_available_started and any(item == "occupied" for item in availability_cache) and len(availability_cache) > 0:
                for i, item in enumerate(availability_cache):
                    if item == "occupied" and to_check[i] == 0:
                        stop_dimsum_thread[i] = False
                        to_check[i] = 1
                        object2[i].add_frame_obj(blank_frame_obj)
                        thread = threading.Thread(target=check_dimsum, args=(i, object2[i]))
                        thread.start()
                        check_dimsum_thread_list[i] = thread
                    elif to_check[i] == 2:
                        object2[i].add_frame_obj(blank_frame_obj)
                    elif to_check[i] == 3:
                        if check_dimsum_thread_list[i].is_alive():
                            check_dimsum_thread_list[i].join()
                    
                    if availability_cache[i] == "unoccupied":
                        to_check[i] = 0
                        realtime_dimsum_found[i] = 0
                        try:
                            if check_dimsum_thread_list[i] != 0 and check_dimsum_thread_list[i].is_alive():
                                stop_dimsum_thread[i] = True
                                check_dimsum_thread_list[i].join()
                        except Exception as e:
                            print("error: ", e)
                            traceback.print_exc()
                            stop_thread = True

            elif check_available_started and all(item == "unoccupied" for item in availability_cache) and len(availability_cache) > 0: #reset
                for i, item in enumerate(availability_cache):
                    to_check[i] = 0
                    realtime_dimsum_found[i] = 0
                    if check_dimsum_thread_list[i] != 0 and check_dimsum_thread_list[i].is_alive():
                        stop_dimsum_thread[i] = True
                        check_dimsum_thread_list[i].join()
            

            if cv2.waitKey(1) == ord("q"):
                stop_thread = True
                end_recording = True
                stop_subprocess.set()
                print('\033[91m' + f'>> [User action] stop all processes' + '\033[0m')
                break
    print('\033[93m' + f'[Main Loop] Ended!' + '\033[0m')

    thread3.join() if thread3.is_alive() else None
    thread2.join() if thread2.is_alive() else None
    thread1.join()  if thread1.is_alive() else None
    if save_preview:
        thread4.join() # When everything done, release the capture
    thread5.join() if thread5.is_alive() else None
    thread6.join() if thread6.is_alive() else None
    thread8.join() if thread8.is_alive() else None
    if simulate:
        fakeCamThread.join()

    print('\033[93m' + f'>> All thread joined' + '\033[0m')

    cap.release()
    cv2.destroyAllWindows()
    print('\033[93m' + f'>> Main function ended' + '\033[0m')
pid = os.getpid()
print("Process ID:", pid)
if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict_update_queue = queue.Queue()
    face_recog_queue = manager.Queue()
    stop_subprocess = multiprocessing.Event()
    shared_dict_update_inprogress = multiprocessing.Event()
    known_employee = manager.dict()

    process = multiprocessing.Process(target=recognize_employee_face, 
                                      args=(shared_dict, face_recog_queue, known_face_encodings, known_face_names, stop_subprocess, known_employee, shared_dict_update_inprogress))
    process.start()


    if config['source'] == "video_frame":
        main(config['source'], simulate, url_path, frame_skipped, start_datetime)

    if config['source'] == "live_frame":
        main(config['source'], False, url_path, 1, datetime.now())

    
    stop_subprocess.set()
    process.join()
    manager.shutdown()
    print('\033[93m' + '>> All subprocess are terminated' + '\033[0m')

    print("ended Process ID:", pid)


