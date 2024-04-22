import pygetwindow as gw
import pyautogui
import cv2
import numpy as np

# Get a list of all windows
windows = gw.getAllWindows()

# Choose the window you want to capture (e.g., the first window)
target_window = windows[8]

# Activate the target window
target_window.activate()

# Get the position and size of the target window
left, top, width, height = target_window.left, target_window.top, target_window.width, target_window.height

# Capture the screen of the target window
screenshot = pyautogui.screenshot(region=(left, top, width, height))

# Convert the screenshot to an OpenCV image
screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Display the screenshot
cv2.imshow("Window Screenshot", screenshot_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
