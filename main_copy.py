import cv2
import numpy as np
from ultralytics import YOLO
from function_bundle import *
import threading
import traceback
import statistics
################ This is old version of main which have imshow the dimsum detected###########
################ Problem: memory leaks

frame_count = 0
frame_rate = 0
stop_thread = False
check_available_started = False
list_total_count_cache = []
list_realtime_count_cache = []
availability_cache = []
end_recording = False
check_dimsum_started = False
var1 = 0
var2 = 0

#### Initialize the color randomizer for detected box ####
detection_colors, class_list = color_selector()

####################### THREADING PROCESS {BEGIN} #######################
def calculate_real_people_total():
    global list_total_count_cache, stop_thread
    #initialize sampling list
    sampling_from_tables = []
    table_name = "customer_events"
    for _ in range (len(table_points)):
        sampling_from_tables.append([])

    try:
        while not stop_thread:
            if check_available_started:
                print("________________-Get in calculate now")
                list_total_count = list_realtime_count_cache.copy()
                
                print("sampling_from_tables", sampling_from_tables)

                # operating
                for i in range (len(table_points)):

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
                for i in range (len(table_points)):
                    if len(sampling_from_tables[i]) > 0: # have sampling data
                        list_total_count[i] = statistics.mode(sampling_from_tables[i])
                        #list_total_count[i] = max(sampling_from_tables[i])
                        #list_total_count[i] = round(sum(sampling_from_tables[i])/len(sampling_from_tables[i]))
                        if (len(sampling_from_tables[i]) == 1 or len(sampling_from_tables[i]) == 25 or len(sampling_from_tables[i]) == 50 or 
                        len(sampling_from_tables[i]) == 75 or len(sampling_from_tables[i]) == 100):
                            #UPDATE customer_events SET customer_amount = %s WHERE 
                            #customer_IN = (SELECT MAX(customer_IN) from customer_events WHERE tableID = %s)
                            update_db(table_name, "customer_amount", list_total_count[i], 
                                    ["customer_IN = (" + select_db("customer_events", ["MAX(customer_IN)"], [f"tableID = {i+1}"]) + ")"])
                            
                    else: # have no sampling data
                        list_total_count[i] = 0
                list_total_count_cache = list_total_count.copy()
                list_total_count.clear()

                time.sleep(5)
    except Exception as e:
        print("error: ", e)
        traceback.print_exc()
        stop_thread = True

def check_available():
    global list_realtime_count_cache, stop_thread, availability_cache, check_available_started
    table_name = "customer_events"
    while not stop_thread:
        table_status = [] # 3 = occupied , 0 = unoccupied
        availability = []
        print("check_available stop_thread = ", stop_thread)
        print("IM INSIDE CHECKAVAILABLE IM INSIDE CHECKAVAILABLE")
        try:
            for table_no, _ in enumerate(table_points): #initialize list
                table_status.append(0)
            # start checking for 3 sampling time at each tables
            for i in range (0, 5):
                print("check available ====================================================", i)
                list_realtime_count = list_realtime_count_cache.copy()
                for table_no, _ in enumerate(table_points):
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
                        insert_db(table_name, field_list, value_list)
                else:                   #unoccupied
                    availability.append("unoccupied")
                    if len(availability_cache) != 0 and availability_cache[i] == "occupied":
                        print("facckkkkk")
                        update_db(table_name, "customer_OUT", datetime.now(), 
                                  ["customer_IN = (" + select_db("customer_events", ["MAX(customer_IN)"], [f"tableID = {i+1}"]) + ")"])
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
    global frame_rate
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
            frame = object1.get_frame()  # Get frame from video queue
            
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

def check_dimsum(table_index, object_frame_in, object_frame_out):
    print("inside check dimsum")
    
    model = YOLO("205trainset_400epoch_model_S.pt")
    global stop_thread, realtime_dimsum_found, check_dimsum_started
    itr = 0
    to_check[table_index] = 2
    try:
        while not stop_thread:
            
            frame_point = table_crop_points[table_index]
            x_min = frame_point[0][0]
            y_min = frame_point[0][1]
            x_max = frame_point[2][0]
            y_max = frame_point[2][1]
            
            if not object_frame_in.is_empty():
                itr += 1
                frame = object_frame_in.get_frame()
                cropped_frame = frame[y_min:y_max, x_min:x_max]
                print("Processed frame from table: ", table_index, "itr = ", itr)
                results = model(source=[cropped_frame], conf=0.6, show=False, save=False, classes=[0])
                annotated_frame = results[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (512,288))
                object_frame_out.add_frame(annotated_frame)
            

            if check_dimsum_started == False:
                check_dimsum_started = True
    except Exception as e:
            print("error: ", e)
            traceback.print_exc()
            stop_thread = True

def combine_frame_dimsum(object_input):
    completed_frames = [None, None, None, None, None, None]
    global stop_thread
    print("GET INSIDE THREAD 5 +++++++++++++++++")
    itr = 0
    try:
        while not stop_thread:
            itr += 1
            # Iterate over each video queue to check for frames
            for index, video_object in enumerate(object_input):
                print("line219")
                print("index = ", index)
                print(video_object.get_len(), "-combine frame [before]-")
                if completed_frames[index] is None:  # Check if frame for this index is not yet filled
                    frame = video_object.get_frame()  # Get frame from video queue
                    if frame is not None:  # If frame is available, update completed_frames
                        completed_frames[index] = frame
                        print("obtained frame of table: ", index)
                        time.sleep(0.1)
            
                print(video_object.get_len(), "-combine frame [after]-\n")
            
            if all(frame is not None for frame in completed_frames):
                print("[----------FRAME USED----------]")
                top_row = cv2.hconcat(completed_frames[0:3])
                bottom_row = cv2.hconcat(completed_frames[3:6])
                combined_frame = cv2.vconcat([top_row, bottom_row])
                completed_frames = [None, None, None, None, None, None]
                combined_frame = cv2.resize(combined_frame, (1920,1080))

                # Display the combined frame
                cv2.imshow('Combined Video', combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.2)
    except Exception as e:
            print("error: ", e)
            traceback.print_exc()


# Create and start the thread
run_event = threading.Event()
run_event.set()



####################### THREADING PROCESS {END} #######################


### load a pretrained YOLOv8n model ###
model = YOLO("weights/yolov8l.pt", "v8")

### select the source of captured frame ###
source = config['source']
if source == "video_frame":
    url_path = config['video_frame']['url']
    cap = cv2.VideoCapture(url_path)

elif source == "live_frame":
    url_path = config['live_frame']['url']
    cap = cv2.VideoCapture(url_path)

### Start reading frame ###
if not cap.isOpened():
    print("Cannot open camera")
    exit()

### Amount of frame skipped ####
frame_skipped = config['frame_skip']

## delete incomplete data with no customer out time before the process
i = "0000-00-00 00:00:00"
condition_list = [f"customer_OUT = '{i}'"]
delete_data_db("customer_events", condition_list)

object1 = videoQueue() #used for thread4 (recording)
object2 = [] #used for detect dimsum
object3 = [] #used for showing dimsum result
check_dimsum_thread_list = []

for i in range (len(table_points)):
    object3.append(videoQueue())

for i in range (len(table_points)):
    obj2 = videoQueue()
    object2.append(obj2)
    check_dimsum_thread_list.append(threading.Thread(target=check_dimsum, args=(i, obj2, object3[i])))

thread1 = threading.Thread(target=calculate_real_people_total)
thread2 = threading.Thread(target=check_available)
thread3 = threading.Thread(target=now_frame_rate)
thread4 = threading.Thread(target=combine_frame)
thread5 = threading.Thread(target=combine_frame_dimsum, args=(object3,))

blank_image = cv2.imread('inference/images/blank.png')
blank_image = cv2.resize(blank_image, (512,288))

while True and not stop_thread:
    # Capture frame-by-frame
    ret, frame = cap.read() # if frame is read correctly ret is True
    
    # Frame counting
    frame_count += 1

    ### Process the frame skipped ###
    if frame_count == 1 or (frame_count) % frame_skipped == 0:
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            stop_thread = True
            end_recording = True
            break
        blank_frame = frame.copy()

        ### Predict on image ###
        detect_params = model(source=[frame], conf=0.4, show=False, save=False, classes=[0], tracker="bytetrack.yaml")

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
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                
                # show center of a person on the frame
                horizon_center = int((int(bb[2]) + int(bb[0]))/2)
                vertical_center = int((int(bb[3]) + int(bb[1]))/2)
                
                cv2.circle(frame, (horizon_center, vertical_center), 10, (0,0,255), -1)

                count_table_people(horizon_center, vertical_center)

                # Display class name and confidence (only used in track mode)
                try:
                    id = int(boxes[i].id.cpu().numpy()[0])
                    cv2.putText(
                        frame,
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
        draw_table_point(frame, availability_cache)
        draw_from_points(frame, table_crop_points)

        preview_queue_length = []
        for item in object3:
            list_length = item.get_len()
            preview_queue_length.append(list_length)

        
        # put text on the bottom right bottom
        text_to_put_list = []
        text_to_put_list.append("frame " + str(frame_count) + " | " + str(frame_rate) + " Frame/s")
        text_to_put_list.append(str(len(detect_params[0])) + " " + "person")
        text_to_put_list.append("realtime: " + str(list_realtime_count))
        text_to_put_list.append("total: " + str(list_total_count_cache))
        text_to_put_list.append(str(availability_cache))
        text_to_put_list.append(str(realtime_dimsum_found))
        text_to_put_list.append(str(preview_queue_length))
        put_text_bottom_right(frame, text_to_put_list)
        print("available: ", availability_cache)
        

        # Display the resulting frame
        cv2.imshow("ObjectDetection", frame)
        #cv2.imwrite("sukrit_restaurant.jpg", frame) 
        list_realtime_count_cache = list_realtime_count.copy()
        reset_people_count()
        
        
        #object1.add_frame(frame) # uncomment without recording cause memory leak!

        if frame_count == 1:
            #thread1.start() #calculate total person
            thread2.start() #check availability
            thread3.start() #check framerate
            #thread4.start() #record video
        # Terminate run when "Q" pressed
        
        if check_available_started and any(item == "occupied" for item in availability_cache) and len(availability_cache) > 0: ## มีบางโต๊ะ occupied (มีคน)
            for i, item in enumerate(availability_cache):
                print("line406")
                print("item: ", i)
                if item == "occupied" and to_check[i] == 0: #if never get into thread, then get into thread
                    print("first condition")
                    var1 += 1
                    to_check[i] = 1
                    object3[i].clear_all()
                    object2[i].add_frame(blank_frame)
                    check_dimsum_thread_list[i].start()
                    print(object3[i].get_len(), "-main-\n")
                    
                if to_check[i] == 2:
                    print("second condition")
                    var1 += 1
                    object2[i].add_frame(blank_frame)
                    print(object3[i].get_len(), "-main-\n")

                if item == "unoccupied" and to_check[i] == 0 and check_dimsum_started:
                    print("Third condition")
                    var2 += 1
                    blank_image = cv2.resize(blank_image, (512,288))
                    object3[i].add_frame(blank_image)
                    print(object3[i].get_len(), "-main-\n")
                
                

            if not thread5.is_alive() and check_dimsum_started:
                thread5.start()
        elif check_available_started and all(item == "unoccupied" for item in availability_cache) and len(availability_cache) > 0:
            if thread5.is_alive():
                thread5.join()

        print("var1 var2 : ", var1, var2)      
        print("to_check: ", to_check)
        print("main - stop_thread = ", stop_thread)
        if cv2.waitKey(1) == ord("q"):
            break

#thread1.join()
thread2.join()
thread3.join()
#thread4.join() # When everything done, release the capture
if thread5.is_alive():
    thread5.join()

cap.release()
cv2.destroyAllWindows()