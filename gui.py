import tkinter as tk
import subprocess
import os

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI for main.py")
        
        self.start_button = tk.Button(self.root, text="Start", command=self.start_main, width=20, height=3)
        self.start_button.pack(pady=50)
        self.start_button.pack(padx=100)
        
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_main, width=20, height=3)
        self.stop_button.pack(pady=50)
        
        self.process = None
        
    def start_main(self):
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.read_stdout()

    def stop_main(self):
        if self.process:
            self.process.terminate()

    def read_stdout(self):
        if self.process:
            output = self.process.stdout.readline().decode("utf-8")
            if output:
                print(output.strip())
                # Do something with the output, like displaying it in a text box or logging
            self.root.after(100, self.read_stdout)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
