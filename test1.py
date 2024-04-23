import pygetwindow as gw
import pyautogui
import cv2
import numpy as np

# Get a list of all windows
windows = gw.getAllWindows()
for index, window in enumerate(windows):
    print(index, window.title)