import threading
import time
import atexit

class MyThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        global frame_count
        second = 1
        while not stop_thread:
            with self.lock:
                frame_count_before = frame_count
            time.sleep(second)
            with self.lock:
                frame_count_after = frame_count
                frame_rate = int((frame_count_after - frame_count_before) / second)
                print("Frame Rate:", frame_rate)
                print(frame_count)

# Global variable
frame_count = 0
stop_thread = False
# Create an instance of MyThread
my_thread = MyThread()
def stop_threads():
    global stop_thread
    stop_thread = True

# Register the function to be called when the program exits
atexit.register(stop_threads)
# Start the thread
my_thread.start()

# Simulate updating frame_count
for _ in range(100):
    time.sleep(0.5)
    with my_thread.lock:
        frame_count += 1
    if frame_count == 50:
        stop_threads()



# Wait for the thread to finish
my_thread.join()

print("Main thread finished")
