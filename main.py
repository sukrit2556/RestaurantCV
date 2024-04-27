import cv2
import numpy as np
from ultralytics import YOLO
from function_bundle import *
import threading
import traceback
import statistics
import queue
import logging
import sys


frame_count = 0
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

#### Initialize the color randomizer for detected box ####
detection_colors, class_list = color_selector()

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
            print("________________-Get in calculate now")
            list_total_count = list_realtime_count_cache.copy()
            
            print("sampling_from_tables", sampling_from_tables)

            target_time = present_datetime + timedelta(seconds=4)

            # operating
            for i in range (len(table_points)):

                print(i)
            
                # if occupied then keep collecting until it reaches 100 collection
                if availability_cache[i] == "occupied" and len(sampling_from_tables[i]) < sampling_amount:
                    realtime_count_that_table = list_total_count[i]
                    sampling_from_tables[i].append(realtime_count_that_table)

                elif availability_cache[i] == "unoccupied": #if become unoccupied then reset
                    sampling_from_tables[i].clear()
                    
                
            print("helloooooooooooooooooooooooooooooooooooooooooooooooooooo")
            print(sampling_from_tables)

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
                                ["customer_ID = (" + f"{select_db('customer_events', ['MAX(customer_ID)'], [f'tableID = {i+1}'])[0]})",
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


def check_available():
    global list_realtime_count_cache, stop_thread, availability_cache, check_available_started, fps, frame_rate

    table_name = "customer_events"
    itr = 0

    while not stop_thread:
        itr += 1
        itr1 = 0
        table_status = [] # 3 = occupied , 0 = unoccupied
        availability = []
        print("check_available stop_thread = ", stop_thread)
        print("IM INSIDE CHECKAVAILABLE IM INSIDE CHECKAVAILABLE")

        try:
            for table_no, _ in enumerate(table_points): #initialize list
                table_status.append(0)
            # start checking for 3 sampling time at each tables
            for i in range (0, 5):
                target_time = present_datetime + timedelta(seconds=4)
                print(present_datetime)
                print(target_time)
                itr1 += 1
                print("check available ====================================================", i)
                list_realtime_count = list_realtime_count_cache.copy()
                for table_no, _ in enumerate(table_points):
                    if list_realtime_count[table_no] > 0: #occupied detected that moment
                        table_status[table_no] += 1
                        print("itr, itr1: ", itr, itr1)
                        print(present_datetime)
                while present_datetime < target_time and not stop_thread:
                    time.sleep(1)
                

            print("table_status", table_status)

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
                                  ["customer_ID = (" + f"{select_db('customer_events', ['MAX(customer_ID)'], [f'tableID = {i+1}'])[0]})",
                                   f"tableID = {i+1}"])
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print(availability)

            availability_cache = availability.copy()
            if check_available_started == False:
                check_available_started = True
            

        except Exception as e:
            print("error: ", e)
            traceback.print_exc()
            stop_thread = True

def now_frame_rate():
    global frame_rate, stop_thread
    second = 1
    while not stop_thread:
        frame_count_before = frame_count
        time.sleep(second)
        frame_count_after = frame_count
        frame_rate = int((frame_count_after - frame_count_before)/second)

def combine_frame():
    completed_frames = None
    global stop_thread
    # Define video resolution and frame rate
    frame_width = 1920
    frame_height = 1080

    size = (frame_width, frame_height) 
    # Define codec and create VideoWriter object
    out = cv2.VideoWriter('result_video/TestOccupied.avi', 
						cv2.VideoWriter_fourcc(*'MJPG'), 
						8, size) 

    while not stop_thread and not end_recording:
        print("COMBINE FRAME IS STILL WORKING FINE")
        # Iterate over each video queue to check for frames
        if completed_frames is None:  # Check if frame for this index is not yet filled
            frame = object1.get_frame()  # Get frame from video queue
            
            if frame is not None:  # If frame is available, update completed_frames
                completed_frames = frame
                
        if frame is not None:
            print(completed_frames)
            completed_frames = cv2.resize(completed_frames, (1920,1080))
            out.write(completed_frames)
            # Display the combined frame
            cv2.imshow('completed_frames', completed_frames)
            completed_frames = None
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #time.sleep(0.1)
    out.release()
    print("out_released")

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
    while not stop_thread:
        if not object_frame_in.is_empty():
            itr += 1
            print("check dimsum table", table_index, "ite = ", itr)
            frame_obj = object_frame_in.get_frame_obj()
            frame = frame_obj.frame
            frame_datetime = frame_obj.date_time
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            results = model(source=[cropped_frame], conf=0.6, show=False, save=False, classes=[0])
            realtime_dimsum_found[table_index] = len(results[0])
            if len(results[0]) > 0:
                found += 1

        #print("table: ", table_index, "itr: ", itr, "found: ", found)
        if itr == 10 and found >= 6:
            to_check[table_index] = 3
            update_db("customer_events", "time_getFood", frame_datetime, 
                      ["customer_ID = (" + f"{select_db('customer_events', ['MAX(customer_ID)'], [f'tableID = {table_index+1}'])[0]})", 
                       f"tableID = {table_index+1}"])
            _, result = select_db("customer_events", ["MAX(customer_ID)"], [f"tableID = {table_index+1}"])
            print(result[0][0])
            object_frame_in.clear_all()
            break
        elif itr == 10 and found < 6:
            itr = 0
            found = 0
        time.sleep(0.1)

def FakeCamera():
    print("inside Fack")

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

# Create and start the thread
run_event = threading.Event()
run_event.set()


####################### THREADING PROCESS {END} #######################
####################### THREADING PROCESS {END} #######################


def main(source_platform, simulate, source_url, frame_skip, date_time):
    ## delete incomplete data with no customer out time before the process

    global stop_thread, frame_count, fps, frame_rate, present_datetime, list_realtime_count_cache, object2, fakeCamFrame, simulate_status, end_recording
    print("fps in main = ", fps)
    ### Start reading frame ###
    if not simulate and  not cap.isOpened():
        print("Cannot open camera")
        exit()
    elif simulate and source_platform == "video_frame":

        # Create a fake camera thread that reads the video in "real-time"
        if frame_count == 0:
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
    
    


    while not stop_thread:
        # Capture frame-by-frame
        if not simulate and source_platform == "video_frame":    #video frame and not simulate fake camera
            ret, frame = cap.read() 
            present_datetime = present_datetime + timedelta(seconds=1/fps)
        elif simulate and source_platform == "video_frame":      #video frame and simulate fake camera
            present_datetime = datetime.now()
            ret, frame = simulate_status, fakeCamFrame.copy()
            """frame = cv2.putText(frame, 
                    "frame " + str(frame_count),
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
        frame_count += 1

        ### Process the frame skipped ###
        if frame_count == 1 or (frame_count) % frame_skip == 0:
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                stop_thread = True
                end_recording = True
                break
            blank_frame = frame_data.copy()
            blank_frame_obj = frame_attr(blank_frame, frame_time)

            ### Predict on image ###
            detect_params = model.track(source=[frame_data], conf=0.4, show=False, save=False, persist=True, classes=[0], tracker="bytetrack.yaml")

            # Convert tensor array to numpy
            DP = detect_params[0].cpu().numpy()
            #print(DP.boxes)
            #print("len", len(detect_params[0]))
            
            if len(DP) != 0:
                for i in range(len(detect_params[0])):

                    boxes = detect_params[0].boxes
                    box = boxes[i]  # returns one box
                    clsID = box.cls.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    bb = box.xyxy.cpu().numpy()[0]

                    cv2.rectangle(
                        frame_data,
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        detection_colors[int(clsID)],
                        3,
                    )

                    font = cv2.FONT_HERSHEY_COMPLEX
                    
                    # show center of a person on the frame
                    horizon_center = int((int(bb[2]) + int(bb[0]))/2)
                    vertical_center = int((int(bb[3]) + int(bb[1]))/2)
                    
                    cv2.circle(frame_data, (horizon_center, vertical_center), 10, (0,0,255), -1)

                    count_table_people(horizon_center, vertical_center)

                    # Display class name and confidence (only used in track mode)
                    try:
                        id = int(boxes[i].id.cpu().numpy()[0])
                        cv2.putText(
                            frame_data,
                            class_list[int(clsID)] + " " + str(id) + " " + str(round(conf, 3)) + "%",
                            (int(bb[0]), int(bb[1]) - 10),
                            font,
                            1,
                            (255, 255, 255),
                            2,
                        )
                    except:
                        pass
            
            ### draw table area ###
            draw_table_point(frame_data, availability_cache)
            draw_from_points(frame_data, table_crop_points)

            
            # put text on the bottom right bottom
            text_to_put_list = []
            text_to_put_list.append("frame " + str(frame_count) + " | " + str(frame_rate) + " Frame/s")
            text_to_put_list.append(str(len(detect_params[0])) + " " + "person")
            text_to_put_list.append("realtime: " + str(list_realtime_count))
            text_to_put_list.append("total: " + str(list_total_count_cache))
            text_to_put_list.append(str(availability_cache))
            text_to_put_list.append("to check" + str(to_check))
            text_to_put_list.append("dimsum found" + str(realtime_dimsum_found))
            text_to_put_list.append("time now: " + str(frame_obj.date_time))
            text_to_put_list.append("fps: " + str(fps))
            put_text_bottom_right(frame_data, text_to_put_list)
            print("available: ", availability_cache)
            

            # Display the resulting frame
            cv2.imshow("ObjectDetection", frame_data)
            #cv2.imwrite("sukrit_restaurant.jpg", frame) 
            list_realtime_count_cache = list_realtime_count.copy()
            reset_people_count()
            
            
            #object1.add_frame(frame) # uncomment without recording cause memory leak!

            if frame_count == 1:
                thread2.start() #check availability
                thread3.start()#check framerate
                #thread4.start() #record video
            # Terminate run when "Q" pressed
            if check_available_started and not thread1.is_alive():
                thread1.start() #calculate total person

            
            if check_available_started and any(item == "occupied" for item in availability_cache) and len(availability_cache) > 0:
                for i, item in enumerate(availability_cache):
                    if item == "occupied" and to_check[i] == 0:
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
                            check_dimsum_thread_list[i] = 0

                    if (to_check[i] == 2 and availability_cache[i] == "unoccupied") or (to_check[i] == 3 and availability_cache[i] == "unoccupied"):
                        to_check[i] = 0
                        realtime_dimsum_found[i] = 0
            elif check_available_started and all(item == "unoccupied" for item in availability_cache) and len(availability_cache) > 0: #reset
                for i, item in enumerate(availability_cache):
                    to_check[i] = 0
                    realtime_dimsum_found[i] = 0
                    if check_dimsum_thread_list[i] != 0 and check_dimsum_thread_list[i].is_alive():
                        check_dimsum_thread_list[i].join()


                    
            print("to_check: ", to_check)
            print("main - stop_thread = ", stop_thread)
            if cv2.waitKey(1) == ord("q"):
                stop_thread = True
                break

    thread3.join()
    thread2.join()
    thread1.join()
    #thread4.join() # When everything done, release the capture

    if simulate:
        fakeCamThread.join()

    cap.release()
    cv2.destroyAllWindows()


if config['source'] == "video_frame":
    main(config['source'], simulate, url_path, frame_skipped, start_datetime)

if config['source'] == "live_frame":
    main(config['source'], False, url_path, 1, datetime.now())