import tkinter as tk
import subprocess

def run_script():
    # Run your Python script here
    subprocess.run(['python', 'main.py'])

# Create the main window
root = tk.Tk()
root.title("Script Runner")

# Create a button to run the script
button = tk.Button(root, text="Run Script", command=run_script)
button.pack(pady=10)

# Run the GUI application
root.mainloop()
