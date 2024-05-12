import cv2
import numpy as np

if __name__ == "__main__":
    from function_bundle import *
    from ultralytics import YOLO
    import multiprocessing
    from encoding_known_face import *
"""if __name__ != "__main__":
    from encoding_known_face import *"""
import threading
import traceback
import statistics
import queue
import logging
import sys
import os
import copy


frame_count = 0
total_frame_count = 0
frame_rate = 0
stop_thread = False
check_available_started = False
list_total_count_cache = []
list_realtime_count_cache = []
availability_cache = []
end_recording = False
check_dimsum_started = False
fakeCamFrame = None
simulate_status = None
update_shared_dict_in_procress = False


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
        print("calculate_real_people_total stopped")

    except Exception as e:
        print("error: ", e)
        traceback.print_exc()
        stop_thread = True


def check_available():
    global list_realtime_count_cache, stop_thread, availability_cache, check_available_started, fps, frame_rate

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
                    if len(availability_cache) == 0 or availability_cache[i] == "unoccupied":
                        field_list = ["tableID", "customer_IN", "created_datetime"]
                        value_list = [i+1, present_datetime, datetime.now()]
                        insert_db(table_name, field_list, value_list)
                else:                   #unoccupied
                    availability.append("unoccupied")
                    if len(availability_cache) != 0 and availability_cache[i] == "occupied":
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

    print("check_available stopped")

def now_frame_rate():
    global frame_rate, stop_thread
    second = 1
    while not stop_thread:
        frame_count_before = frame_count
        time.sleep(second)
        frame_count_after = frame_count
        frame_rate = int((frame_count_after - frame_count_before)/second)
    print("now_frame_rate stopped")

def combine_frame():
    completed_frames = None
    global stop_thread
    # Define video resolution and frame rate
    frame_width = 1920
    frame_height = 1080

    size = (frame_width, frame_height) 
    # Define codec and create VideoWriter object
    out = cv2.VideoWriter('result_video/processing_record.mp4', 
                            cv2.VideoWriter_fourcc(*'H264'), 
                            7, size) 

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

def check_dimsum(table_index, object_frame_in):
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
    print("Check_dimsum stopped")

def FakeCamera():

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
    print("fake_camera stopped")

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
                        #select id from customer_event where created_datetime = (max(created_datetim) from table=i+1) and table = i+1
                        _, customerID = select_db("customer_events", ["customer_ID"], 
                                                ["created_datetime = (" + f"{select_db('customer_events', ['MAX(created_datetime)'], [f'tableID = {i+1}'])[0]})", f"tableID = {i+1}"])
                        image_full_file_path = get_media_abs_path(customerID[0][0], "wrapped_up.mp4")
                        abs_path[i] = image_full_file_path      #save absolute path dir to video to list
                        image_relate_file_path = get_media_relate_path(customerID[0][0], "wrapped_up.mp4")
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
        print("record_customer_activity stopped")
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
            update_shared_dict_in_procress = True
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
            
            update_shared_dict_in_procress = False
    print("update_shared_dict stopped")

def recognize_employee_face(shared_dict, todo_queue, known_face_encodings, known_face_names, stop_subprocess):
    
    import face_recognition
    import dlib
    import time
    #wait until update_shared_dict is started
    #while update_shared_dict == False:
        #sleep
    try:
        while not stop_subprocess.is_set():
            filtered_dict = {}
            while todo_queue.empty() and not stop_subprocess.is_set():
                #print("subprocess waiting ...")
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

                #key_to_change = []

                #face recog part >>>
                # Pop items until the dictionary is empty
                while filtered_dict:
                    key, value = filtered_dict.popitem()
                    #print(f"subprocess recognizing key {key} ")
                    top_left = value.top_left
                    bottom_right = value.bottom_right
                    y_min = top_left[1]
                    y_max = bottom_right[1]
                    x_min = top_left[0]
                    x_max = bottom_right[0]
                    cropped_person = frame[y_min:y_max, x_min:x_max]
                    cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)

                    cv2.imwrite("fuckthis.jpg", cropped_person)
                    #print(f"stop subprocess = {stop_subprocess.is_set()}")
                    #find face location and encoding
                    face_location = face_recognition.face_locations(cropped_person, model='hog')

                    face_encodings = face_recognition.face_encodings(cropped_person, face_location)

                    #face comparing
                    for face_encoding in face_encodings:
                        # Compare face encoding with known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # If a match is found, use the known face name
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]

                            #update shared dict part >>>
                            if key in shared_dict:
                                # Update the value of the key
                                shared_dict[key].person_type = name
                                shared_dict[key].fixed = True
                                print(f"shared_dict[key].person_type = {shared_dict[key].person_type} fix = {shared_dict[key].fixed}")
                                print("Updated shared_dict['{}']".format(key))
                                print('\033[91m' + 'I recognized someone!!!' + '\033[0m')
                                cv2.imwrite("foundyou.jpg", cropped_person)

                                break
        print("Sub process is out of touch")
    except Exception as e:
        print("error: ", e)
        traceback.print_exc()
        stop_subprocess.set()


####################### THREADING PROCESS {END} #######################
####################### THREADING PROCESS {END} #######################


def main(source_platform, simulate, source_url, frame_skip, date_time):
    ## delete incomplete data with no customer out time before the process

    global stop_thread, frame_count, fps, frame_rate, present_datetime, list_realtime_count_cache, object2, fakeCamFrame, simulate_status, end_recording
    global blank_frame_cache, total_frame_count

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


    i = "0000-00-00 00:00:00"
    condition_list = [f"customer_OUT = '{i}'"]
    delete_data_db("customer_events", condition_list)

    thread1 = threading.Thread(target=calculate_real_people_total)
    thread2 = threading.Thread(target=check_available)
    thread3 = threading.Thread(target=now_frame_rate)
    thread4 = threading.Thread(target=combine_frame)
    thread5 = threading.Thread(target=record_customer_activities)
    thread6 = threading.Thread(target=update_shared_dict)
    
    


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


        frame_obj = frame_attr(frame, present_datetime)
        frame_data = frame_obj.frame
        frame_time = frame_obj.date_time
        
        # Frame counting
        total_frame_count += 1

        ### Process the frame skipped ###
        if total_frame_count == 1 or (total_frame_count) % frame_skip == 0:
            if not ret or stop_subprocess.is_set():
                print("Can't receive frame (stream end?). Exiting ...")
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
                
                while update_shared_dict_in_procress:
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
                    #print('\033[91m' + 'to classify unknown_customer' + '\033[0m')
                    #if person is customer
                    classify_unknown_customer(local_dict, id, is_customer, frame_count, present_datetime, 
                                              (horizon_center, vertical_center),(int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),)
                    
                    key_contain_in_frame.append(id)

                    # Display class name and confidence (only used in track mode)
                    try:
                        id = int(boxes[i].id.cpu().numpy()[0])
                        for pt1, pt2 in local_dict[id].all_edge():
                            cv2.line(frame_data, pt1, pt2, 
                                     detection_colors[int(0 if local_dict[id].person_type == "customer" else (1 if local_dict[id].person_type == "customer" else 2))], thickness=4, lineType=cv2.LINE_AA)
                        cv2.rectangle(
                            frame_data,
                            (int(bb[0]), int(bb[1])),
                            (int(bb[2]), int(bb[3])),
                            detection_colors[int(0 if local_dict[id].person_type == "customer" else (1 if local_dict[id].person_type == "customer" else 2))],
                            3,
                        )
                        cv2.rectangle(frame_data, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[1]+30)), 
                                      detection_colors[int(0 if local_dict[id].person_type == "customer" else (1 if local_dict[id].person_type == "customer" else 2))], -1) 
                        cv2.putText(
                            frame_data,
                            class_list[int(clsID)] + " " + str(id),
                            (int(bb[0]), int(bb[1]) + 25),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )
                        cv2.putText(
                            frame_data,
                            str(local_dict[id].person_type) + " " + "{:.2f}".format(local_dict[id].probToBeCustomer),
                            (int(bb[0]), int(bb[1]) + 50),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )
                        """cv2.putText(
                            frame_data,
                            str(local_dict[id].dt_first_found) + " " + str(local_dict[id].dt_latest_found),
                            (int(bb[0]), int(bb[1]) + 75),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )
                        cv2.putText(
                            frame_data,
                            str(local_dict[id].frame_first_found)+ " " + str(local_dict[id].frame_latest_found),
                            (int(bb[0]), int(bb[1]) + 100),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )"""
                        cv2.putText(
                            frame_data,
                            str(local_dict[id].fixed),
                            (int(bb[0]), int(bb[1]) + 75),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )
                        """cv2.putText(
                            frame_data,
                            str(local_dict[id].top_left) + " " + str(local_dict[id].bottom_right),
                            (int(bb[0]), int(bb[1]) + 100),
                            font,
                            1,
                            (255, 255, 255),
                            1,
                        )"""
                    except Exception as e:
                        print("error: ", e)
                        traceback.print_exc()
                        stop_thread = True
            #remove key that don't have now
            if len(key_contain_in_frame) > 0:
                update_local_dict(local_dict, key_contain_in_frame, frame_count)
                shared_dict_update_queue.put(local_dict)
            

            ### draw table area ###
            draw_table_point(frame_data, availability_cache)
            #draw_from_points(frame_data, table_crop_points, (255, 0, 0))
            #draw_from_points(frame_data, plotted_points_recording, (0, 255, 255))

            
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
            reset_people_count()
            
            
            #object1.add_frame(frame_data) # uncomment without recording cause memory leak!

            if frame_count == 1:
                thread2.start() #check availability
                thread3.start()#check framerate
                #thread4.start() #record video
                thread6.start()
            # Terminate run when "Q" pressed
            if check_available_started and not thread1.is_alive():
                thread1.start() #calculate total person
                thread5.start()

            
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
                break
    print("done loop")

    thread3.join() if thread3.is_alive() else None
    thread2.join() if thread2.is_alive() else None
    thread1.join()  if thread1.is_alive() else None
    #thread4.join() # When everything done, release the capture
    thread5.join() if thread5.is_alive() else None
    thread6.join() if thread6.is_alive() else None
    print("after join thread")
    if simulate:
        fakeCamThread.join()

    print("after stop simulate")

    cap.release()
    print("715")
    cv2.destroyAllWindows()
    print("719")

    print("main program ended")
pid = os.getpid()
print("Process ID:", pid)
if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict_update_queue = queue.Queue()
    face_recog_queue = manager.Queue()
    stop_subprocess = multiprocessing.Event()

    process = multiprocessing.Process(target=recognize_employee_face, 
                                      args=(shared_dict, face_recog_queue, known_face_encodings, known_face_names, stop_subprocess))
    process.start()


    if config['source'] == "video_frame":
        main(config['source'], simulate, url_path, frame_skipped, start_datetime)

    if config['source'] == "live_frame":
        main(config['source'], False, url_path, 1, datetime.now())

    print(shared_dict.keys())
    
    stop_subprocess.set()
    process.join()
    manager.shutdown()
    print("ended Process ID:", pid)

    print("after clear face_recog queue")
    print("after kill queue")
    print(face_recog_queue)
    print(shared_dict)
