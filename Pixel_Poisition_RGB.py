import pyautogui
from PIL import ImageGrab
import time

try:
    while True:
        # Get and print the current mouse cursor position
        x, y = pyautogui.position()
        print(f"Mouse Cursor Position - X: {x}, Y: {y}")

        # grab the small pixel color 
        # image.grab(left, top, right, bottom)
        screenshot = ImageGrab.grab(bbox=(x, y, x + 1, y + 1)) #grab the small pixel bounding box
        rgb_values = screenshot.getpixel((0, 0))
        print(f"RGB Values - R: {rgb_values[0]}, G: {rgb_values[1]}, B: {rgb_values[2]}")

        # Optional: Add a short delay to control the loop frequency
        time.sleep(3)  # Adjust the sleep duration as needed

except KeyboardInterrupt:
    print("Loop interrupted by the user.")
