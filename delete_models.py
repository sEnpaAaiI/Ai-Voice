import sys
import os
import time
import threading
import shutil
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

logging.basicConfig(level=logging.DEBUG)

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
        except Exception as e:
            logging.error(f"Error in models_folder loop: {e}")

        try:
            audios_folder = os.path.join(now_dir, "audios")

            for element in os.listdir(audios_folder):
                element_route = os.path.join(audios_folder, element)
                if os.path.isfile(element_route):
                    os.remove(element_route)
        except Exception as e:
            logging.error(f"Error in audios_folder loop: {e}")

        wait_time = 24 * 60 * 60
        logging.info(f"Sleeping for {wait_time} seconds")
        time.sleep(wait_time)

def start_infinite_loop():
    hilo_bucle = threading.Thread(target=infinite_loop)
    hilo_bucle.daemon = True
    hilo_bucle.start()