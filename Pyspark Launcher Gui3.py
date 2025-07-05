import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
import os
import sys
import shutil

print("目前使用的 Python 路徑：", sys.executable)

# Java 11 路徑（Gephi 附帶的）
JAVA_HOME = r"C:\Program Files\Gephi-0.10.1\jre-x64\jdk-11.0.17+8-jre"
# 指定 Python 3.10 可執行檔
PYTHON_EXE = r"C:\Users\Leon\AppData\Local\Programs\Python\Python310\python.exe"
# Hadoop winutils 根目錄與執行檔
HADOOP_HOME = r"C:\Winutils"
WINUTILS_BIN = os.path.join(HADOOP_HOME, 'bin')
WINUTILS_EXE = os.path.join(WINUTILS_BIN, 'winutils.exe')
# Spark checkpoint 目錄
CHECKPOINT_DIR = r"C:\tmp\spark-checkpoint"

selected_script = None

def select_script():
    global selected_script
    path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
    if path:
        selected_script = path
        status_label.config(text=f"選擇腳本：{os.path.basename(path)}")

def run_pyspark():
    if not selected_script:
        messagebox.showwarning("提醒", "請先選擇 .py 腳本")
        return

    env = os.environ.copy()
    # 設定 Java 環境
    env['JAVA_HOME'] = JAVA_HOME
    env['PATH'] = os.path.join(JAVA_HOME, 'bin') + os.pathsep + env['PATH']
    # 設定 HADOOP_HOME 與 winutils
    env['HADOOP_HOME'] = HADOOP_HOME
    env['PATH'] = WINUTILS_BIN + os.pathsep + env['PATH']
    # 使用指定 Python
    env['PYSPARK_PYTHON'] = PYTHON_EXE
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)

    # 確保目錄存在
    os.makedirs(WINUTILS_BIN, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # 複製 winutils.exe
    src_winutils = os.path.join(HADOOP_HOME, 'winutils.exe')
    if os.path.exists(src_winutils) and not os.path.exists(WINUTILS_EXE):
        try:
            shutil.copy(src_winutils, WINUTILS_EXE)
        except Exception as e:
            print(f"無法複製 winutils.exe：{e}")

    try:
        # 顯示 Java 版本
        java_cmd = os.path.join(JAVA_HOME, 'bin', 'java.exe')
        subprocess.run([java_cmd, '-version'], env=env)

        # 測試 winutils
        if os.path.isfile(WINUTILS_EXE):
            subprocess.run([WINUTILS_EXE, 'ls'], env=env)
        else:
            print(f"警告: 無法找到 winutils.exe：{WINUTILS_EXE}")

        # 執行 PySpark 腳本，並把 checkpoint 目錄傳入環境
        env['SPARK_CHECKPOINT_DIR'] = CHECKPOINT_DIR
        subprocess.Popen([
            PYTHON_EXE,
            selected_script
        ], env=env, cwd=os.path.dirname(selected_script)).communicate()

    except Exception as e:
        messagebox.showerror("錯誤", f"執行失敗：{e}")

# 建立 GUI
root = tk.Tk()
root.title('簡易 PySpark 啟動器')
root.geometry('500x200')

tk.Label(root, text='選擇要執行的 PySpark 腳本 (.py)').pack(pady=10)
select_btn = tk.Button(root, text='選擇腳本', command=select_script, bg='lightblue')
select_btn.pack()
status_label = tk.Label(root, text='尚未選擇腳本', fg='gray')
status_label.pack(pady=5)

run_btn = tk.Button(root, text='執行 PySpark', command=run_pyspark, bg='lightgreen')
run_btn.pack(pady=15)

root.mainloop()
