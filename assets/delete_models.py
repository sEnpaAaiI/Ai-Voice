import sys
import os
import time
import threading
import shutil

now_dir = os.getcwd()
sys.path.append(now_dir)


def infinite_loop():
    while True:
        try:
            models_folder = os.path.join(now_dir, "logs")

            for element in os.listdir(models_folder):
                element_route = os.path.join(models_folder, element)
                if os.path.isdir(element_route) and element != "mute":
                    shutil.rmtree(element_route)
                elif os.path.isfile(element_route):
                    os.remove(element_route)

            wait_time = 24 * 60 * 60  # 
            time.sleep(wait_time)
        except:
            pass

def start_infinite_loop():
    hilo_bucle = threading.Thread(target=infinite_loop)
    hilo_bucle.start()