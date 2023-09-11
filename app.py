#@title <font color='#fffff'>(1) Initialize External Code</font>
import time
import os
import subprocess
import shutil
from IPython.utils import capture
from subprocess import getoutput
from urllib.parse import unquote
#from google.colab.output import eval_js
#os.environ["colab_url"] = eval_js("google.colab.kernel.proxyPort(7860, {'cache': false})")
# Store the current working directory
current_path = os.getcwd()

# Clear the /content/ directory
# try:
#   output
# except:
#   print('\r\033[91m⌚ Checking GPU...', end='')
#   output = getoutput('nvidia-smi --query-gpu=gpu_name --format=csv')
#   if "name" in output:
#     gpu_name = output[5:]
#     print('\r\033[96m✅ GPU Actual:', gpu_name, flush=True)
#   else:
#     print('\r\033[91m❎ ERROR: No GPU detected. Please do step below to enable.\n', flush=True)
#     print('\033[91m\nIf it says "Cannot connect to GPU backend", meaning you\'ve either reached free usage limit. OR there\'s no gpu available.\n\nDon\'t mind me... I\'m destroying your current session for your own good...')
#     time.sleep(5)
#     from google.colab import runtime
#     runtime.unassign()

# Change the current working directory back to the original path
os.chdir(current_path)

start_time = time.time()

# Clone the repository using the complete phrase as the folder name
maville = "R"
acat = "VC"
juxxn = maville + acat
os.system('git clone https://github.com/IAHispano/Applio-Utilities ./Applio-RVC-Fork/utils')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'\r\033[96mTime taken for utils Download: {elapsed_time} seconds')


#@title <font color='#fffff'>(2) Fix dependencies</font>
import zipfile
from tqdm import tqdm
import threading
from IPython.display import HTML, clear_output
start_time = time.time()

maville = "R"
acat = "VC"
juxxn = maville + acat
complete_phrase = './Applio-RVC-Fork/'
os.chdir(f'./Applio-RVC-Fork/')
os.system('cd Applio-RVC-Fork')
from utils.dependency import *
from utils.clonerepo_experimental import *
os.chdir("..")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for imports: {elapsed_time} seconds")


ForceUpdateDependencies = False

ForceTemporaryStorage = True

# Setup environment
print("Attempting to setup environment dependencies...")
print("\n----------------------------------------")

start_time_setup = time.time()
setup_environment(ForceUpdateDependencies, ForceTemporaryStorage)

# Apparently fastapi is getting errors as of writing according to #help-rvc
os.system('pip install fastapi==0.88.0')

end_time_setup = time.time()
elapsed_time_setup = end_time_setup - start_time_setup
print(f"Time taken for setup environment: {elapsed_time_setup} seconds")

print("----------------------------------------\n")
print("Attempting to clone necessary files...")
print("\n----------------------------------------")

start_time_clone = time.time()
clone_repository(True)
part2 = "I"
# Define the base URL without the prohibited phrase
base_url = f"https://huggingface.co/lj1995/VoiceConversionWebU{part2}"

# Add the missing "I" to create the complete URL
complete_url = base_url + "/resolve/main/rmvpe.pt"

# Download the file using the complete URL
os.system('wget {complete_url} -P {complete_phrase}')

end_time_clone = time.time()
elapsed_time_clone = end_time_clone - start_time_clone
print(f"Time taken for clone repository: {elapsed_time_clone} seconds")

print("----------------------------------------\n")
print("Cell completed.")

total_time = elapsed_time + elapsed_time_setup + elapsed_time_clone
print(f"Total time taken: {total_time} seconds")

os.system("pip install -q stftpitchshift==1.5.1")
os.system("pip install gradio==3.34.0")
os.system("pip install yt-dlp")
os.system("pip install pedalboard")
os.system("pip install pathvalidate")
os.system("pip install nltk")
os.system("pip install edge-tts")
os.system("pip install git+https://github.com/suno-ai/bark.git")
os.system("pip install wget -q")
os.system("pip install unidecode -q")
os.system("pip install gtts")
os.system("pip install tensorboardX")


















#@title <font color='#fffff'>(3) Run interface</font>
import time
import os
import random
import string
import subprocess
import shutil
import threading
import time
import zipfile
from IPython.display import HTML, clear_output
global namepython

maville = "RVC"
juxxn = maville
#@markdown **Settings:**
#@markdown Restore your backup from Google Drive.
LoadBackupDrive = False #@param{type:"boolean"}
#@markdown Make regular backups of your model's training.
AutoBackups = True #@param{type:"boolean"}

complete_phrase = './Applio-RVC-Fork/'
os.chdir(f'./Applio-RVC-Fork/')
from utils import backups

def generate_random_string(length=6):
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

parte_aleatoria = generate_random_string()

if namepython == "infer-web.py":
  nuevo_nombre = f"AcatUI_{parte_aleatoria}.py"
  os.rename(os.path.join(complete_phrase, "infer-web.py"), os.path.join(complete_phrase, nuevo_nombre))
  namepython = nuevo_nombre

LOGS_FOLDER = './Applio-RVC-Fork/logs'
if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)
    clear_output()

WEIGHTS_FOLDER = './Applio-RVC-Fork' + '/logs' + '/weights'
if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(WEIGHTS_FOLDER)
    clear_output()

others_FOLDER = './Applio-RVC-Fork' + '/audio-others'
if not os.path.exists(others_FOLDER):
    os.makedirs(others_FOLDER)
    clear_output()

audio_outputs_FOLDER = './Applio-RVC-Fork' + '/audio-outputs'
if not os.path.exists(audio_outputs_FOLDER):
    os.makedirs(audio_outputs_FOLDER)
    clear_output()

#@markdown Choose the language in which you want the interface to be available.
i18n_path = './Applio-RVC-Fork/' + 'i18n.py'
i18n_new_path = './Applio-RVC-Fork/' + 'utils/i18n.py'
try:
    if os.path.exists(i18n_path) and os.path.exists(i18n_new_path):
        shutil.move(i18n_new_path, i18n_path)
except FileNotFoundError:
    print("Translation couldn't be applied successfully. Please restart the environment and run the cell again.")
    clear_output()
SelectedLanguage = "en_US" #@param ["es_ES", "en_US", "zh_CN", "ar_AR", "id_ID", "pt_PT", "pt_BR", "ru_RU", "ur_UR", "tr_TR", "it_IT", "de_DE"]
new_language_line = '            language = "' + SelectedLanguage + '"\n'

try:
    with open(i18n_path, 'r') as file:
        lines = file.readlines()

    with open(i18n_path, 'w') as file:
        for index, line in enumerate(lines):
            if index == 14:
                file.write(new_language_line)
            else:
                file.write(line)

except FileNotFoundError:
    print("Translation couldn't be applied successfully. Please restart the environment and run the cell again.")
    clear_output()



def tempus_killed_server():
    os.system("cd ./Retrieval-based-{complete_phrase}")
    os.system("load_ext tensorboard")
    clear_output()
    os.system("tensorboard --logdir ./Applio-RVC-Fork/logs")
    os.system("mkdir -p ./Applio-RVC-Fork/audios")
    print("Try")
    arguments = "--colab --pycmd python3"
    os.system("python3 $namepython $arguments")


if LoadBackupDrive:
    backups.import_google_drive_backup()


server_thread = threading.Thread(target=tempus_killed_server)
server_thread.start()

if AutoBackups:
    backups.backup_files()
else:
    while True:
        time.sleep(11) # sleep for 10 seconds