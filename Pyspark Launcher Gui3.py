#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡易 PySpark 啟動器（固定使用 Python 3.10）
──────────────────────────────────────────────────────────────
1. 選程式  (.py)
2. 選資料  (.csv)
3. 執行   :  <Python-3.10 exe> <script> --csv <data>
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess, os, sys

# ── 強制指定 Python 3.10 ──────────────────────────────────────────────────
PYTHON_EXE = r"C:\Users\Leon\AppData\Local\Programs\Python\Python310\python.exe"
print("將使用的 Python 路徑：", PYTHON_EXE)

# ── 如果想保留偵錯訊息，仍輸出目前啟動 GUI 的解譯器 ───────────────
print("啟動 GUI 的解譯器路徑：", sys.executable)

# ── Java11 (Gephi 內附) ───────────────────────────────────────────────────
JAVA_HOME = r"C:\Program Files\Gephi-0.10.1\jre-x64\jdk-11.0.17+8-jre"

selected_script: str | None = None
selected_data:   str | None = None

# ──────────────────────────────────────────────────────────────
def select_file():
    global selected_script
    path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
    if path:
        selected_script = path
        status_label.config(text=f"已選程式: {os.path.basename(path)}")

def select_csv():
    global selected_data
    path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if path:
        selected_data = path
        csv_label.config(text=f"已選資料: {os.path.basename(path)}")

def run_pyspark():
    if not selected_script:
        messagebox.showwarning("提醒", "請先選擇 .py 腳本")
        return
    if not selected_data:
        messagebox.showwarning("提醒", "請先選擇 CSV 資料檔")
        return

    env = os.environ.copy()
    env["JAVA_HOME"]       = JAVA_HOME
    env["PATH"]            = JAVA_HOME + r"\bin;" + env["PATH"]
    env["PYSPARK_PYTHON"]  = PYTHON_EXE          # Spark Worker 也用 3.10
    env.pop("PYTHONHOME", None); env.pop("PYTHONPATH", None)

    print("\n[Script Output]")
    try:
        subprocess.run(
            [PYTHON_EXE, selected_script, "--csv", selected_data],
            env=env, check=True
        )
    except subprocess.CalledProcessError as e:
        messagebox.showerror("錯誤", f"腳本執行失敗\n{e}")

# ── Tkinter UI ───────────────────────────────────────────────
root = tk.Tk()
root.title("PySpark TD-UF 啟動器 (Py 3.10)")
root.geometry("520x260")

tk.Label(root, text="1️⃣  選擇主程式 (.py)").pack(pady=6)
tk.Button(root, text="選擇程式", command=select_file,
          bg="lightblue").pack()
status_label = tk.Label(root, text="尚未選擇程式", fg="gray")
status_label.pack()

tk.Label(root, text="2️⃣  選擇資料檔 (.csv)").pack(pady=6)
tk.Button(root, text="選擇資料檔 (CSV)", command=select_csv,
          bg="lightyellow").pack()
csv_label = tk.Label(root, text="尚未選擇資料檔", fg="gray")
csv_label.pack()

tk.Button(root, text="🚀  執行 PySpark", command=run_pyspark,
          bg="lightgreen", width=18, height=2).pack(pady=14)

root.mainloop()
