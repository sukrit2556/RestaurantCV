import cv2
import sys
import time
import logging
import numpy as np
import threading, queue

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
fps = 0
# This is shared between main and the FakeCamera
currentFrame = None

def FakeCamera(Q, filename):

    global fps
    """Reads the video file at its natural rate, storing the frame in a global called 'currentFrame'"""
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

    # Initialise currentFrame
    global currentFrame
    currentFrame = np.zeros((h,w,3), dtype=np.uint8)

    # Signal main that we are ready
    Q.put('OK')

    while True:
        ret,frame = video.read()
        if ret == False:
            break
        # Store video frame where main can access it
        currentFrame[:] = frame[:]
        # Try and read at appropriate rate
        time.sleep(1.0/fps)

    logging.debug('[FakeCamera] Ending')
    Q.put('DONE')

if __name__ == '__main__':

    #  Create a queue for synchronising and communicating with our fake camera
    Q = queue.Queue()

    # Create a fake camera thread that reads the video in "real-time"
    fc = threading.Thread(target=FakeCamera, args=(Q,'inference/videos/occupied_test_full.mp4'))
    fc.start()

    # Wait for fake camera to intialise
    logging.debug(f'[main] Waiting for camera to power up and initialise')
    msg = Q.get()
    if msg != 'OK':
        sys.exit()

    # Main processing loop should go here - we'll just grab a couple frames at different times
    while True:
    # Check if there's a message available
        if not Q.empty():
            msg = Q.get()
            # Break the loop if the message indicates the end of the video
            if msg == 'DONE':
                break

        # Show the current frame
        cv2.imshow('Video', currentFrame)
        res = cv2.waitKey(int(1000/fps))
    
    cv2.destroyAllWindows()
    # Wait for buddy to finish
    fc.join()