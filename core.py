import os
import sys
import json
import argparse
import subprocess
import spaces

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config import Config

from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

from rvc.infer.infer import infer_pipeline

from rvc.lib.tools.model_download import model_download_pipeline

config = Config()
current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

# Get TTS Voices
with open(os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r") as f:
    voices_data = json.load(f)

locales = list({voice["Locale"] for voice in voices_data})


# Infer
@spaces.GPU
def run_infer_script(
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    input_path,
    output_path,
    pth_path,
    index_path,
    split_audio,
    f0autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
):
    f0autotune = "True" if str(f0autotune) == "True" else "False"
    clean_audio = "True" if str(clean_audio) == "True" else "False"
    upscale_audio = "True" if str(upscale_audio) == "True" else "False"
    infer_pipeline(
        f0up_key,
        filter_radius,
        index_rate,
        rms_mix_rate,
        protect,
        hop_length,
        f0method,
        input_path,
        output_path,
        pth_path,
        index_path,
        split_audio,
        f0autotune,
        clean_audio,
        clean_strength,
        export_format,
        embedder_model,
        embedder_model_custom,
        upscale_audio,
    )
    return f"File {input_path} inferred successfully.", output_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Batch infer
@spaces.GPU
def run_batch_infer_script(
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    input_folder,
    output_folder,
    pth_path,
    index_path,
    split_audio,
    f0autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
):
    f0autotune = "True" if str(f0autotune) == "True" else "False"
    clean_audio = "True" if str(clean_audio) == "True" else "False"
    upscale_audio = "True" if str(upscale_audio) == "True" else "False"
    audio_files = [
        f for f in os.listdir(input_folder) if f.endswith((".mp3", ".wav", ".flac"))
    ]
    print(f"Detected {len(audio_files)} audio files for inference.")

    for audio_file in audio_files:
        if "_output" in audio_file:
            pass
        else:
            input_path = os.path.join(input_folder, audio_file)
            output_file_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(
                output_folder,
                f"{output_file_name}_output{os.path.splitext(audio_file)[1]}",
            )
            print(f"Inferring {input_path}...")

            infer_pipeline(
                f0up_key,
                filter_radius,
                index_rate,
                rms_mix_rate,
                protect,
                hop_length,
                f0method,
                input_path,
                output_path,
                pth_path,
                index_path,
                split_audio,
                f0autotune,
                clean_audio,
                clean_strength,
                export_format,
                embedder_model,
                embedder_model_custom,
                upscale_audio,
            )

    return f"Files from {input_folder} inferred successfully."


# TTS
@spaces.GPU
def run_tts_script(
    tts_text,
    tts_voice,
    tts_rate,
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    output_tts_path,
    output_rvc_path,
    pth_path,
    index_path,
    split_audio,
    f0autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
):
    f0autotune = "True" if str(f0autotune) == "True" else "False"
    clean_audio = "True" if str(clean_audio) == "True" else "False"
    upscale_audio = "True" if str(upscale_audio) == "True" else "False"
    tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")

    if os.path.exists(output_tts_path):
        os.remove(output_tts_path)

    command_tts = [
        "python",
        tts_script_path,
        tts_text,
        tts_voice,
        str(tts_rate),
        output_tts_path,
    ]
    subprocess.run(command_tts)

    infer_pipeline(
        f0up_key,
        filter_radius,
        index_rate,
        rms_mix_rate,
        protect,
        hop_length,
        f0method,
        output_tts_path,
        output_rvc_path,
        pth_path,
        index_path,
        split_audio,
        f0autotune,
        clean_audio,
        clean_strength,
        export_format,
        embedder_model,
        embedder_model_custom,
        upscale_audio,
    )

    return f"Text {tts_text} synthesized successfully.", output_rvc_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Download
def run_download_script(model_link):
    model_download_pipeline(model_link)
    return f"Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(pretraineds_v1, pretraineds_v2, models, exe):
    prequisites_download_pipeline(pretraineds_v1, pretraineds_v2, models, exe)
    return "Prerequisites installed successfully."

# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the main.py script with specific parameters."
    )
    subparsers = parser.add_subparsers(
        title="subcommands", dest="mode", help="Choose a mode"
    )

    # Parser for 'infer' mode
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--f0up_key",
        type=str,
        help="Value for f0up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    infer_parser.add_argument(
        "--filter_radius",
        type=str,
        help="Value for filter_radius",
        choices=[str(i) for i in range(11)],
        default="3",
    )
    infer_parser.add_argument(
        "--index_rate",
        type=str,
        help="Value for index_rate",
        choices=[str(i / 10) for i in range(11)],
        default="0.3",
    )
    infer_parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    infer_parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    infer_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    infer_parser.add_argument(
        "--f0method",
        type=str,
        help="Value for f0method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    infer_parser.add_argument("--input_path", type=str, help="Input path")
    infer_parser.add_argument("--output_path", type=str, help="Output path")
    infer_parser.add_argument("--pth_path", type=str, help="Path to the .pth file")
    infer_parser.add_argument(
        "--index_path",
        type=str,
        help="Path to the .index file",
    )
    infer_parser.add_argument(
        "--split_audio",
        type=str,
        help="Enable split audio",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--f0autotune",
        type=str,
        help="Enable autotune",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--clean_audio",
        type=str,
        help="Enable clean audio",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--clean_strength",
        type=str,
        help="Value for clean_strength",
        choices=[str(i / 10) for i in range(11)],
        default="0.7",
    )
    infer_parser.add_argument(
        "--export_format",
        type=str,
        help="Export format",
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=["contentvec", "hubert", "custom"],
        default="hubert",
    )
    infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )
    infer_parser.add_argument(
        "--upscale_audio",
        type=str,
        help="Enable audio upscaling",
        choices=["True", "False"],
        default="False",
    )

    # Parser for 'batch_infer' mode
    batch_infer_parser = subparsers.add_parser(
        "batch_infer", help="Run batch inference"
    )
    batch_infer_parser.add_argument(
        "--f0up_key",
        type=str,
        help="Value for f0up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    batch_infer_parser.add_argument(
        "--filter_radius",
        type=str,
        help="Value for filter_radius",
        choices=[str(i) for i in range(11)],
        default="3",
    )
    batch_infer_parser.add_argument(
        "--index_rate",
        type=str,
        help="Value for index_rate",
        choices=[str(i / 10) for i in range(11)],
        default="0.3",
    )
    batch_infer_parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    batch_infer_parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    batch_infer_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    batch_infer_parser.add_argument(
        "--f0method",
        type=str,
        help="Value for f0method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    batch_infer_parser.add_argument("--input_folder", type=str, help="Input folder")
    batch_infer_parser.add_argument("--output_folder", type=str, help="Output folder")
    batch_infer_parser.add_argument(
        "--pth_path", type=str, help="Path to the .pth file"
    )
    batch_infer_parser.add_argument(
        "--index_path",
        type=str,
        help="Path to the .index file",
    )
    batch_infer_parser.add_argument(
        "--split_audio",
        type=str,
        help="Enable split audio",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--f0autotune",
        type=str,
        help="Enable autotune",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--clean_audio",
        type=str,
        help="Enable clean audio",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--clean_strength",
        type=str,
        help="Value for clean_strength",
        choices=[str(i / 10) for i in range(11)],
        default="0.7",
    )
    batch_infer_parser.add_argument(
        "--export_format",
        type=str,
        help="Export format",
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    batch_infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=["contentvec", "hubert", "custom"],
        default="hubert",
    )
    batch_infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )
    batch_infer_parser.add_argument(
        "--upscale_audio",
        type=str,
        help="Enable audio upscaling",
        choices=["True", "False"],
        default="False",
    )

    # Parser for 'tts' mode
    tts_parser = subparsers.add_parser("tts", help="Run TTS")
    tts_parser.add_argument(
        "--tts_text",
        type=str,
        help="Text to be synthesized",
    )
    tts_parser.add_argument(
        "--tts_voice",
        type=str,
        help="Voice to be used",
        choices=locales,
    )
    tts_parser.add_argument(
        "--tts_rate",
        type=str,
        help="Increase or decrease TTS speed",
        choices=[str(i) for i in range(-100, 100)],
        default="0",
    )
    tts_parser.add_argument(
        "--f0up_key",
        type=str,
        help="Value for f0up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    tts_parser.add_argument(
        "--filter_radius",
        type=str,
        help="Value for filter_radius",
        choices=[str(i) for i in range(11)],
        default="3",
    )
    tts_parser.add_argument(
        "--index_rate",
        type=str,
        help="Value for index_rate",
        choices=[str(i / 10) for i in range(11)],
        default="0.3",
    )
    tts_parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    tts_parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    tts_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    tts_parser.add_argument(
        "--f0method",
        type=str,
        help="Value for f0method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    tts_parser.add_argument("--output_tts_path", type=str, help="Output tts path")
    tts_parser.add_argument("--output_rvc_path", type=str, help="Output rvc path")
    tts_parser.add_argument("--pth_path", type=str, help="Path to the .pth file")
    tts_parser.add_argument(
        "--index_path",
        type=str,
        help="Path to the .index file",
    )
    tts_parser.add_argument(
        "--split_audio",
        type=str,
        help="Enable split audio",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--f0autotune",
        type=str,
        help="Enable autotune",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--clean_audio",
        type=str,
        help="Enable clean audio",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--clean_strength",
        type=str,
        help="Value for clean_strength",
        choices=[str(i / 10) for i in range(11)],
        default="0.7",
    )
    tts_parser.add_argument(
        "--export_format",
        type=str,
        help="Export format",
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    tts_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=["contentvec", "hubert", "custom"],
        default="hubert",
    )
    tts_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )
    tts_parser.add_argument(
        "--upscale_audio",
        type=str,
        help="Enable audio upscaling",
        choices=["True", "False"],
        default="False",
    )

    # Parser for 'download' mode
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument(
        "--model_link",
        type=str,
        help="Link of the model",
    )

    # Parser for 'prerequisites' mode
    prerequisites_parser = subparsers.add_parser(
        "prerequisites", help="Install prerequisites"
    )
    prerequisites_parser.add_argument(
        "--pretraineds_v1",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Download pretrained models for v1",
    )
    prerequisites_parser.add_argument(
        "--pretraineds_v2",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Download pretrained models for v2",
    )
    prerequisites_parser.add_argument(
        "--models",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Donwload models",
    )
    prerequisites_parser.add_argument(
        "--exe",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Download executables",
    )

    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        print("Please run the script with '-h' for more information.")
        sys.exit(1)

    args = parse_arguments()

    try:
        if args.mode == "infer":
            run_infer_script(
                str(args.f0up_key),
                str(args.filter_radius),
                str(args.index_rate),
                str(args.rms_mix_rate),
                str(args.protect),
                str(args.hop_length),
                str(args.f0method),
                str(args.input_path),
                str(args.output_path),
                str(args.pth_path),
                str(args.index_path),
                str(args.split_audio),
                str(args.f0autotune),
                str(args.clean_audio),
                str(args.clean_strength),
                str(args.export_format),
                str(args.embedder_model),
                str(args.embedder_model_custom),
                str(args.upscale_audio),
            )
        elif args.mode == "batch_infer":
            run_batch_infer_script(
                str(args.f0up_key),
                str(args.filter_radius),
                str(args.index_rate),
                str(args.rms_mix_rate),
                str(args.protect),
                str(args.hop_length),
                str(args.f0method),
                str(args.input_folder),
                str(args.output_folder),
                str(args.pth_path),
                str(args.index_path),
                str(args.split_audio),
                str(args.f0autotune),
                str(args.clean_audio),
                str(args.clean_strength),
                str(args.export_format),
                str(args.embedder_model),
                str(args.embedder_model_custom),
                str(args.upscale_audio),
            )
        elif args.mode == "tts":
            run_tts_script(
                str(args.tts_text),
                str(args.tts_voice),
                str(args.tts_rate),
                str(args.f0up_key),
                str(args.filter_radius),
                str(args.index_rate),
                str(args.rms_mix_rate),
                str(args.protect),
                str(args.hop_length),
                str(args.f0method),
                str(args.output_tts_path),
                str(args.output_rvc_path),
                str(args.pth_path),
                str(args.index_path),
                str(args.split_audio),
                str(args.f0autotune),
                str(args.clean_audio),
                str(args.clean_strength),
                str(args.export_format),
                str(args.embedder_model),
                str(args.embedder_model_custom),
                str(args.upscale_audio),
            )
        elif args.mode == "download":
            run_download_script(
                str(args.model_link),
            )
        elif args.mode == "prerequisites":
            run_prerequisites_script(
                str(args.pretraineds_v1),
                str(args.pretraineds_v2),
                str(args.models),
                str(args.exe),
            )
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
