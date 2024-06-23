import gradio as gr
import sys
import os
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

# Tabs
from tabs.inference.inference import inference_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab

# Assets
import assets.themes.loadThemes as loadThemes
from assets.i18n.i18n import I18nAuto
import assets.installation_checker as installation_checker
from assets.discord_presence import RPCManager
from assets.flask.server import start_flask, load_config_flask
from core import run_prerequisites_script
from delete_models import start_infinite_loop

run_prerequisites_script("False", "True", "True", "True")
start_infinite_loop()

i18n = I18nAuto()
installation_checker.check_installation()
logging.getLogger("uvicorn").disabled = True
logging.getLogger("fairseq").disabled = True

my_applio = loadThemes.load_json()
if my_applio:
    pass
else:
    my_applio = "ParityError/Interstellar"

with gr.Blocks(theme=my_applio, title="Applio") as Applio:
    gr.Markdown("# Applio")
    gr.Markdown("### From the first Applio to the last")
    gr.Markdown(
        i18n(
            "Ultimate voice cloning tool, meticulously optimized for unrivaled power, modularity, and user-friendly experience."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/IAHispano) — [Discord Bot](https://discord.com/oauth2/authorize?client_id=1144714449563955302&permissions=1376674695271&scope=bot%20applications.commands) — [Find Voices](https://applio.org/models) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Download")):
        download_tab()


def launch_gradio():
    Applio.launch()


if __name__ == "__main__":
    launch_gradio()