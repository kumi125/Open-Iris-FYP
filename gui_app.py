import tkinter as tk
from tkinter import messagebox
import os
import glob

from preprocess_iris import preprocess_latest
from segment_iris import segment_iris
from main import compare_images  # reuse your function

USER_DB = "data/users"


# -----------------------------
# REGISTER FUNCTION
# -----------------------------
def register_user():
    username = entry_username.get().strip()

    if username == "":
        messagebox.showerror("Error", "Enter username")
        return

    user_path = os.path.join(USER_DB, username)
    os.makedirs(user_path, exist_ok=True)

    status_label.config(text="Capturing 3 samples...")

    for i in range(3):
        status_label.config(text=f"Capture {i+1}/3 → Press S then Q")
        root.update()

        os.system("python test_camera.py")
        preprocess_latest()
        segment_iris(user_path)

    messagebox.showinfo("Success", f"{username} registered!")


# -----------------------------
# LOGIN FUNCTION
# -----------------------------
def login_user():
    username = entry_username.get().strip()
    user_path = os.path.join(USER_DB, username)

    if not os.path.exists(user_path):
        messagebox.showerror("Error", "User not found")
        return

    status_label.config(text="Capturing login image...")
    root.update()

    os.system("python test_camera.py")
    preprocess_latest()

    temp_folder = "data/temp_login"
    os.makedirs(temp_folder, exist_ok=True)

    test_iris = segment_iris(temp_folder)

    if test_iris is None:
        messagebox.showerror("Error", "Segmentation failed")
        return

    user_iris = glob.glob(os.path.join(user_path, "*.jpg"))

    scores = []

    for ref in user_iris:
        score = compare_images(ref, test_iris)
        scores.append(score)

    min_score = min(scores)

    # 🔥 FINAL DECISION
    if min_score < 0.15:
        messagebox.showinfo("Login", f"✅ Welcome {username}")
    elif min_score < 0.20:
        messagebox.showwarning("Login", "⚠️ Uncertain, try again")
    else:
        messagebox.showerror("Login", "❌ Access Denied")


# -----------------------------
# UI DESIGN
# -----------------------------
root = tk.Tk()
root.title("Iris Recognition System")
root.geometry("400x300")
root.configure(bg="#1e1e2f")

# Title
title = tk.Label(root, text="IRIS AUTH SYSTEM", font=("Arial", 18, "bold"), fg="white", bg="#1e1e2f")
title.pack(pady=15)

# Username input
entry_username = tk.Entry(root, font=("Arial", 12), width=25)
entry_username.pack(pady=10)

# Buttons
btn_register = tk.Button(root, text="Register", font=("Arial", 12), bg="#4CAF50", fg="white", width=15, command=register_user)
btn_register.pack(pady=5)

btn_login = tk.Button(root, text="Login", font=("Arial", 12), bg="#2196F3", fg="white", width=15, command=login_user)
btn_login.pack(pady=5)

# Status
status_label = tk.Label(root, text="", font=("Arial", 10), fg="yellow", bg="#1e1e2f")
status_label.pack(pady=20)

root.mainloop()