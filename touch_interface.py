import tkinter as tk
import subprocess

def run_ai_script():
    subprocess.run(["python3", "AI_script.py"])


root = tk.Tk()
root.title("Sound Probe")
root.geometry("300x200")


probe_button = tk.Button(root, text="Probe", font=("Arial", 24), command=run_ai_script)
probe_button.pack(expand=True)


root.mainloop()
