import cv2
import sys
import time
import logging
import numpy as np
import threading, queue
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')



fakeCamFrame = None
fakeCam_started = False
fps = 0  #### uncomment this if you want to run on this file

def FakeCamera(Q, filename):

    global fps, fakeCam_started
    """Reads the video file at its natural rate, storing the frame in a global called 'fakeCamFrame'"""
    logging.debug(f'[FakeCamera] Generating video stream from {filename}')

    # Open video
    video = cv2.VideoCapture(filename)
    if (video.isOpened()== False):
       logging.critical(f'[FakeCamera] Unable to open video {filename}')
       Q.put('ERROR')
       return

    # Get height, width and framerate so we know how often to read a frame
    h   = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    logging.debug(f'[FakeCamera] h={h}, w={w}, fps={fps}')

    # Initialise fakeCamFrame
    global fakeCamFrame
    fakeCamFrame = np.zeros((h,w,3), dtype=np.uint8)
    print(fps)

    # Signal main that we are ready
    Q.put(True)

    while True:
        ret, frame = video.read()
        if ret == False:
            break
        # Store video frame where main can access it
        fakeCamFrame[:] = frame[:]
        # Try and read at appropriate rate
        fakeCam_started = True
        time.sleep(1.0/fps)

    logging.debug('[FakeCamera] Ending')
    Q.put(False)

if __name__ == '__main__':
    frame_count = 0
    simulate = True
    source = "video_frame"
    start_position_x = 500
    start_position_y = 650
    font = cv2.FONT_HERSHEY_COMPLEX
    #  Create a queue for synchronising and communicating with our fake camera
    if simulate and source == "video_frame":
        Q = queue.Queue()

        # Create a fake camera thread that reads the video in "real-time"
        if frame_count == 0:
            fc = threading.Thread(target=FakeCamera, args=(Q,'inference/videos/restaurantLong.mp4'))
            fc.start()

        # Wait for fake camera to intialise
        logging.debug(f'[main] Waiting for camera to power up and initialise')
        ret = Q.get()
        if ret != True:
            sys.exit()

    # Main processing loop should go here - we'll just grab a couple frames at different times
    while True:
    # Check if there's a message available
        if not Q.empty() and fakeCam_started == True:
            ret = Q.get()
            # Break the loop if the message indicates the end of the video
            if ret == False:
                break
        frame_count += 1
        frame = fakeCamFrame.copy()
        frame = cv2.putText(frame, 
                "frame " + str(frame_count),
                (start_position_x,start_position_y),
                font,       #font name
                1,          #font scale
                (0,0,255),  #font color
                2           #font thickness
        )
        # Show the current frame
        cv2.imshow('fakevid', frame)
        
        if cv2.waitKey(1) == ord("q"):
                break
    
    cv2.destroyAllWindows()
    # Wait for buddy to finish
    fc.join()