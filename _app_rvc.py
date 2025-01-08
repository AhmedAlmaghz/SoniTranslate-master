import gradio as gr
from typing import List
from soni_translate.translate_segments import (
    translate_and_cache_segments,
    get_translate_from_cache,
)
from soni_translate.text_to_speech import (
    generate_speech_for_text,
    generate_speech_for_segments,
)
from soni_translate.audio_segments import create_audio_segments
from soni_translate.speech_segmentation import create_segments_from_speech
from soni_translate.text_multiformat_processor import (
    process_text_to_segments,
    process_segments_to_text,
)
from soni_translate.preprocessor import audio_video_preprocessor, audio_preprocessor
from soni_translate.postprocessor import (
    merge_audio_segments,
    merge_speech_segments,
    merge_translated_segments,
)
from soni_translate.utils import (
    get_valid_files,
    extract_video_links,
    get_link_list,
    download_list,
    is_audio_file,
    is_video_file,
    is_subtitle_file,
    remove_files,
    create_directories,
    copy_files,
    rename_file,
)
from soni_translate.language_configuration import (
    load_language_names,
    load_language_defaults,
    get_language_code,
    set_language_defaults,
)
from soni_translate.languages_gui import (
    load_languages_gui,
    get_news_gui,
    get_tutorial_gui,
)
from soni_translate.logging_setup import logger
from soni_translate.version import VERSION
from voice_main import ClassVoices, BASE_DIR, BASE_MODELS, BASE_DOWNLOAD_LINK
from lib.rmvpe import download_rmvpe
from lib.audio import remix_audio
from lib.i18n import I18nAuto
from types import SimpleNamespace
import os
import glob
import json
import uuid
import shutil
import subprocess
import time
import re
import asyncio
import edge_tts
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import psutil
import platform
import sys
from soni_translate.youtube_upload import upload_video_to_youtube

i18n = I18nAuto()

# Read configuration from environment variables or set default values
TEMP_FILE = os.environ.get("TEMP_FILE", "TEMP")
PITCH_EXTRACTION_MODEL = os.environ.get(
    "PITCH_EXTRACTION_MODEL", "rmvpe.pt"
).strip()
PITCH_DETECTION_RANGE = [
    int(os.environ.get("PITCH_DETECTION_RANGE_MIN", "50")),
    int(os.environ.get("PITCH_DETECTION_RANGE_MAX", "1100")),
]
INDEX_RATE = float(os.environ.get("INDEX_RATE", "0.75"))
FILTER_RADIUS = int(os.environ.get("FILTER_RADIUS", "3"))
PITCH_CHANGE = int(os.environ.get("PITCH_CHANGE", "0"))
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "Unconditional")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "auto")
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))
ENVELOPE_MIX = float(os.environ.get("ENVELOPE_MIX", "0.25"))
PROTECT_VOICELESS = float(os.environ.get("PROTECT_VOICELESS", "0.33"))
REMIX_OPTION = os.environ.get("REMIX_OPTION", "rebalance")
REMIX_EXTRA = float(os.environ.get("REMIX_EXTRA", "0"))
REMIX_VOLUME = float(os.environ.get("REMIX_VOLUME", "0"))
REMIX_CENTER = float(os.environ.get("REMIX_CENTER", "0"))
ADJUST_VOLUMES = os.environ.get("ADJUST_VOLUMES", "original")
AUDIO_MIXING_METHOD = os.environ.get("AUDIO_MIXING_METHOD", "none")
SUBTITLE_TYPE = os.environ.get("SUBTITLE_TYPE", "soft")
CONVERT_NUMBERS = os.environ.get("CONVERT_NUMBERS", "true") == "true"
CLEAN_AUDIO = os.environ.get("CLEAN_AUDIO", "true") == "true"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))
MAX_VRAM_WORKERS = int(os.environ.get("MAX_VRAM_WORKERS", "1"))
MAX_CPU_WORKERS = int(os.environ.get("MAX_CPU_WORKERS", "1"))
MAX_VRAM = int(os.environ.get("MAX_VRAM", "0"))
MAX_RAM = int(os.environ.get("MAX_RAM", "0"))
MAX_DURATION = int(os.environ.get("MAX_DURATION", "120"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "15"))
SEGMENT_SIZE = int(os.environ.get("SEGMENT_SIZE", "45"))
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3")
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "google_translator_batch")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
USE_CUDA = os.environ.get("USE_CUDA", "true") == "true"
USE_PREVIEW = os.environ.get("USE_PREVIEW", "false") == "true"
ONLY_VOICE = os.environ.get("ONLY_VOICE", "false") == "true"
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "mp4")
USE_API = os.environ.get("USE_API", "false") == "true"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CUSTOM_MODEL = os.environ.get("CUSTOM_MODEL", "")
CUSTOM_VOICE = os.environ.get("CUSTOM_VOICE", "")
CUSTOM_INDEX = os.environ.get("CUSTOM_INDEX", "")
CUSTOM_PITCH = os.environ.get("CUSTOM_PITCH", "")
CUSTOM_TRANSLATOR = os.environ.get("CUSTOM_TRANSLATOR", "")
CUSTOM_TTS = os.environ.get("CUSTOM_TTS", "")
CUSTOM_TTS_SPEAKER = os.environ.get("CUSTOM_TTS_SPEAKER", "")
CUSTOM_TTS_LANGUAGE = os.environ.get("CUSTOM_TTS_LANGUAGE", "")
CUSTOM_TTS_SPEED = float(os.environ.get("CUSTOM_TTS_SPEED", "1.0"))
CUSTOM_ASR = os.environ.get("CUSTOM_ASR", "")
CUSTOM_ASR_OPTIONS = os.environ.get("CUSTOM_ASR_OPTIONS", "")
CUSTOM_PROCESSOR = os.environ.get("CUSTOM_PROCESSOR", "")
CUSTOM_POSTPROCESSOR = os.environ.get("CUSTOM_POSTPROCESSOR", "")
CUSTOM_DOWNLOADS = os.environ.get("CUSTOM_DOWNLOADS", "")
CUSTOM_EXTRA_OPTIONS = os.environ.get("CUSTOM_EXTRA_OPTIONS", "")

# Set default values for language codes
(
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    TTS_LANGUAGE,
    WHISPER_LANGUAGE,
    TRANSLATE_FROM_CLIP,
    TRANSLATE_TO_CLIP,
) = load_language_defaults()

# Load language names for supported languages
LANGUAGE_NAMES = load_language_names()

# Load language-specific GUI text
(
    news_gui,
    tutorial_gui,
    language_news,
    language_tutorial,
    language_codes,
    language_dict,
) = load_languages_gui(TARGET_LANGUAGE)

# Initialize the model
model_voice = ClassVoices(only_cpu=not USE_CUDA)

# Set the maximum number of threads for OpenMP
os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_WORKERS)

# Set the CUDA device
if USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set environment variables for paths
os.environ["MODELS_DIR"] = os.environ.get(
    "MODELS_DIR", os.path.join(BASE_DIR, "models")
)
os.environ["WAV_DIR"] = os.environ.get("WAV_DIR", os.path.join(BASE_DIR, "audios"))
os.environ["VIDEO_DIR"] = os.environ.get(
    "VIDEO_DIR", os.path.join(BASE_DIR, "videos")
)
os.environ["SUBTITLE_DIR"] = os.environ.get(
    "SUBTITLE_DIR", os.path.join(BASE_DIR, "subtitles")
)
os.environ["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", os.path.join(BASE_DIR, "outputs")
)

# Create directories if they don't exist
create_directories(
    [
        os.environ["WAV_DIR"],
        os.environ["VIDEO_DIR"],
        os.environ["SUBTITLE_DIR"],
        os.environ["OUTPUT_DIR"],
    ]
)

# Download required models
for model in BASE_MODELS:
    download_link = os.path.join(BASE_DOWNLOAD_LINK, model)
    subprocess.run(
        [
            "python",
            "-m",
            "soni_translate.utils",
            "--download_manager",
            download_link,
            BASE_DIR,
        ]
    )

# Download RMVPE model if it doesn't exist
if not os.path.exists(PITCH_EXTRACTION_MODEL):
    download_rmvpe(BASE_DIR, PITCH_EXTRACTION_MODEL)

# Download custom models
if CUSTOM_DOWNLOADS:
    download_list(CUSTOM_DOWNLOADS)

# Load TTS voices
tts_voices = []
try:
    tts_voices = [voice.name for voice in edge_tts.list_voices()]
except:
    logger.warning("No internet connection to load tts voices")

# Load custom models
custom_models = []
if os.path.exists("custom_models.json"):
    with open("custom_models.json", "r") as f:
        custom_models = json.load(f)

# Load custom voices
custom_voices = []
if os.path.exists("custom_voices.json"):
    with open("custom_voices.json", "r") as f:
        custom_voices = json.load(f)

# Load custom index files
custom_index_files = []
if os.path.exists("custom_index.json"):
    with open("custom_index.json", "r") as f:
        custom_index_files = json.load(f)

# Load custom pitch files
custom_pitch_files = []
if os.path.exists("custom_pitch.json"):
    with open("custom_pitch.json", "r") as f:
        custom_pitch_files = json.load(f)

# Load custom translators
custom_translators = []
if os.path.exists("custom_translators.json"):
    with open("custom_translators.json", "r") as f:
        custom_translators = json.load(f)

# Load custom TTS
custom_tts_list = []
if os.path.exists("custom_tts.json"):
    with open("custom_tts.json", "r") as f:
        custom_tts_list = json.load(f)

# Load custom ASR models
custom_asr_models = []
if os.path.exists("custom_asr.json"):
    with open("custom_asr.json", "r") as f:
        custom_asr_models = json.load(f)

# Load custom processors
custom_processors = []
if os.path.exists("custom_processors.json"):
    with open("custom_processors.json", "r") as f:
        custom_processors = json.load(f)

# Load custom postprocessors
custom_postprocessors = []
if os.path.exists("custom_postprocessors.json"):
    with open("custom_postprocessors.json", "r") as f:
        custom_postprocessors = json.load(f)

# Load custom ASR options
custom_asr_options = []
if os.path.exists("custom_asr_options.json"):
    with open("custom_asr_options.json", "r") as f:
        custom_asr_options = json.load(f)

# Load custom extra options
custom_extra_options = []
if os.path.exists("custom_extra_options.json"):
    with open("custom_extra_options.json", "r") as f:
        custom_extra_options = json.load(f)

# Create a thread pool for processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Function to check for updates
def check_for_updates():
    try:
        # Get the latest release information from GitHub
        response = subprocess.check_output(
            ["git", "ls-remote", "https://github.com/RVC-Project/SoniTranslate.git"],
            stderr=subprocess.STDOUT,
        )
        latest_commit = response.decode("utf-8").split("\n")[0].split("\t")[0]

        # Get the current commit hash
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode("utf-8").strip()

        # Compare the current commit with the latest commit
        if latest_commit != current_commit:
            return "Update available! Please run 'git pull' to get the latest version."
        else:
            return "You are using the latest version."
    except Exception as e:
        return f"Error checking for updates: {e}"

# Function to get system information
def get_system_info():
    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.machine(),
        "Processor": platform.processor(),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Logical Cores": psutil.cpu_count(logical=True),
        "Total RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        "Available RAM": f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
        "Total Disk Space": f"{psutil.disk_usage('/').total / (1024 ** 3):.2f} GB",
        "Used Disk Space": f"{psutil.disk_usage('/').used / (1024 ** 3):.2f} GB",
        "Free Disk Space": f"{psutil.disk_usage('/').free / (1024 ** 3):.2f} GB",
        "GPU Memory": "Not Available",
    }

    try:
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            system_info["GPU"] = gpu_info.name
            system_info[
                "GPU Memory"
            ] = f"{gpu_info.total_memory / (1024 ** 3):.2f} GB"
        else:
            system_info["GPU"] = "No GPU found or CUDA not available"
    except Exception as e:
        system_info["GPU"] = f"Error getting GPU info: {e}"

    return system_info

# Function to reload GUI text
def reload_gui_text():
    global news_gui, tutorial_gui, language_news, language_tutorial, language_codes, language_dict
    (
        news_gui,
        tutorial_gui,
        language_news,
        language_tutorial,
        language_codes,
        language_dict,
    ) = load_languages_gui(TARGET_LANGUAGE)
    return (
        news_gui,
        tutorial_gui,
        gr.Dropdown.update(choices=language_codes, value=TARGET_LANGUAGE),
    )

# Function to set language defaults
def set_defaults(
    source_language,
    target_language,
    tts_language,
    whisper_language,
    translate_from_clip,
    translate_to_clip,
):
    set_language_defaults(
        source_language,
        target_language,
        tts_language,
        whisper_language,
        translate_from_clip,
        translate_to_clip,
    )
    return (
        source_language,
        target_language,
        tts_language,
        whisper_language,
        translate_from_clip,
        translate_to_clip,
    )

# Function to get news
def get_news():
    return get_news_gui(language_news)

# Function to get tutorial
def get_tutorial():
    return get_tutorial_gui(language_tutorial)
from soni_translate.youtube_upload import upload_video_to_youtube

# Function to process files
def process_files(
    operation_mode,
    input_path,
    output_path,
    source_language,
    target_language,
    tts_speaker,
    tts_language,
    tts_speed,
    whisper_language,
    translate_from_clip,
    translate_to_clip,
    pitch_change,
    index_rate,
    filter_radius,
    envelope_mix,
    protect_voiceless,
    remix_option,
    remix_extra,
    remix_volume,
    remix_center,
    adjust_volumes,
    audio_mixing_method,
    subtitle_type,
    convert_numbers,
    clean_audio,
    max_duration,
    batch_size,
    chunk_size,
    segment_size,
    asr_model,
    translate_model,
    compute_type,
    use_preview,
    only_voice,
    output_format,
    hf_token,
    custom_model,
    custom_voice,
    custom_index,
    custom_pitch,
    custom_translator,
    custom_tts,
    custom_tts_speaker,
    custom_tts_language,
    custom_tts_speed,
    custom_asr,
    custom_asr_options,
    custom_processor,
    custom_postprocessor,
    custom_extra_options,
    youtube_channel_id,
    youtube_token,
    progress=gr.Progress(track_tqdm=True),
):
    # Check for updates
    update_message = check_for_updates()
    logger.info(update_message)

    # Get system information
    system_info = get_system_info()
    logger.info("System Information:")
    for key, value in system_info.items():
        logger.info(f"{key}: {value}")

    # Validate input parameters
    if not input_path:
        raise gr.Error("Input path is required.")
    if not output_path:
        output_path = os.environ["OUTPUT_DIR"]

    # Get a list of valid files based on the operation mode
    input_files = get_valid_files(
        extract_video_links(input_path) if operation_mode == "link" else [input_path]
    )

    # Process each input file
    for input_file in input_files:
        # Generate a unique identifier for the input file
        file_id = str(uuid.uuid4())

        # Create temporary directories for processing
        temp_dir = os.path.join(TEMP_FILE, file_id)
        create_directories([temp_dir])

        # Determine the output file path based on the input file type
        if is_audio_file(input_file):
            output_file = os.path.join(
                output_path, os.path.basename(input_file) + f".{output_format}"
            )
        elif is_video_file(input_file):
            output_file = os.path.join(
                output_path, os.path.basename(input_file) + f".{output_format}"
            )
        elif is_subtitle_file(input_file):
            output_file = os.path.join(
                output_path, os.path.basename(input_file) + ".txt"
            )
        else:
            raise gr.Error(
                "Invalid input file type. Only audio, video, and subtitle files are supported."
            )

        # Copy the input file to the temporary directory
        copy_files(input_file, temp_dir)

        # Get the base name of the input file
        base_name = os.path.basename(input_file)

        # Get the paths of the audio, video, and subtitle files in the temporary directory
        audio_files, video_files, subtitle_files = get_directory_files(temp_dir)

        # Determine the appropriate preprocessing function based on the input file type
        preprocess_func = (
            audio_video_preprocessor if video_files else audio_preprocessor
        )

        # Preprocess the input file
        if preprocess_func == audio_video_preprocessor:
            input_file_path = video_files[0]
        else:
            input_file_path = audio_files[0]

        preprocess_func(
            use_preview,
            input_file_path,
            os.path.join(temp_dir, "Video.mp4"),
            os.path.join(temp_dir, "audio.wav"),
            use_cuda=USE_CUDA,
        )

        # Get the paths of the audio, video, and subtitle files after preprocessing
        audio_files, video_files, subtitle_files = get_directory_files(temp_dir)

        # Process audio files
        if audio_files:
            for audio_file in audio_files:
                # Create audio segments
                segments = create_audio_segments(
                    audio_file,
                    temp_dir,
                    max_duration,
                    chunk_size,
                    batch_size,
                    compute_type,
                    asr_model,
                    whisper_language,
                    hf_token,
                    custom_asr,
                    custom_asr_options,
                    progress,
                )

                # Translate audio segments
                translated_segments = translate_and_cache_segments(
                    segments,
                    translate_model,
                    source_language,
                    target_language,
                    translate_from_clip,
                    translate_to_clip,
                    custom_translator,
                    progress,
                )

                # Generate speech for translated segments
                if operation_mode != "translate":
                    translated_segments = generate_speech_for_segments(
                        translated_segments,
                        tts_speaker,
                        tts_language,
                        tts_speed,
                        custom_tts,
                        custom_tts_speaker,
                        custom_tts_language,
                        custom_tts_speed,
                        progress,
                    )

                # Merge translated segments
                merged_file = merge_translated_segments(
                    translated_segments,
                    audio_file,
                    subtitle_type,
                    convert_numbers,
                    custom_postprocessor,
                )

                # Clean audio if enabled
                if clean_audio:
                    merged_file = remix_audio(
                        merged_file,
                        remix_option,
                        remix_volume,
                        remix_center,
                        remix_extra,
                        adjust_volumes,
                        progress,
                    )

                # Copy the merged file to the output directory
                copy_files(merged_file, output_path)

        # Process video files
        if video_files:
            for video_file in video_files:
                # Create speech segments from the video
                speech_segments = create_segments_from_speech(
                    video_file,
                    temp_dir,
                    max_duration,
                    segment_size,
                    compute_type,
                    asr_model,
                    whisper_language,
                    hf_token,
                    custom_asr,
                    custom_asr_options,
                    progress,
                )

                # Translate speech segments
                translated_segments = translate_and_cache_segments(
                    speech_segments,
                    translate_model,
                    source_language,
                    target_language,
                    translate_from_clip,
                    translate_to_clip,
                    custom_translator,
                    progress,
                )

                # Generate speech for translated segments
                if operation_mode != "translate":
                    translated_segments = generate_speech_for_segments(
                        translated_segments,
                        tts_speaker,
                        tts_language,
                        tts_speed,
                        custom_tts,
                        custom_tts_speaker,
                        custom_tts_language,
                        custom_tts_speed,
                        progress,
                    )

                # Merge speech segments
                merged_file = merge_speech_segments(
                    translated_segments,
                    video_file,
                    audio_mixing_method,
                    subtitle_type,
                    convert_numbers,
                    custom_postprocessor,
                )

                # Clean audio if enabled
                if clean_audio:
                    merged_file = remix_audio(
                        merged_file,
                        remix_option,
                        remix_volume,
                        remix_center,
                        remix_extra,
                        adjust_volumes,
                        progress,
                    )

                # Copy the merged file to the output directory
                copy_files(merged_file, output_path)

                # Upload the merged file to YouTube if a token is provided
                if youtube_token:
                    try:
                        upload_video_to_youtube(
                            merged_file,
                            youtube_channel_id,
                            youtube_token,
                            "Translated Video",  # Default title
                            "This video has been translated using SoniTranslate.",  # Default description
                            ["SoniTranslate", "translation", "AI"],  # Default tags
                        )
                        logger.info(f"Video uploaded to YouTube: {youtube_channel_id}")
                    except Exception as e:
                        logger.error(f"Error uploading video to YouTube: {e}")

        # Process subtitle files
        if subtitle_files:
            for subtitle_file in subtitle_files:
                # Process text to segments
                segments = process_text_to_segments(
                    subtitle_file,
                    max_duration,
                    chunk_size,
                    custom_processor,
                )

                # Translate segments
                translated_segments = translate_and_cache_segments(
                    segments,
                    translate_model,
                    source_language,
                    target_language,
                    translate_from_clip,
                    translate_to_clip,
                    custom_translator,
                    progress,
                )

                # Generate speech for translated segments
                if operation_mode != "translate":
                    translated_segments = generate_speech_for_segments(
                        translated_segments,
                        tts_speaker,
                        tts_language,
                        tts_speed,
                        custom_tts,
                        custom_tts_speaker,
                        custom_tts_language,
                        custom_tts_speed,
                        progress,
                    )

                # Merge translated segments
                merged_file = merge_translated_segments(
                    translated_segments,
                    subtitle_file,
                    subtitle_type,
                    convert_numbers,
                    custom_postprocessor,
                )

                # Copy the merged file to the output directory
                copy_files(merged_file, output_path)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

    return output_file, output_file

# Create the Gradio interface
with gr.Blocks(
    css="footer {visibility: hidden}",
    title="SoniTranslate",
    analytics_enabled=False,
    theme=gr.themes.Soft(),
) as app:
    # Set the language code for the interface
    language_code = gr.State(TARGET_LANGUAGE)

    # Create a header for the interface
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"<center><h1><a href='https://github.com/RVC-Project/SoniTranslate'>SoniTranslate {VERSION}</a></h1></center>"
            )
            gr.Markdown(
                f"<center><h3>{i18n('Free and open source audio, video and text translator, with voice cloning and support for large files.')}</h3></center>"
            )
        with gr.Column():
            logo = os.path.join(os.path.dirname(__file__), "assets", "logo.jpeg")
            gr.Image(
                value=logo,
                label="logo",
                width=100,
                height=100,
                interactive=False,
                container=False,
            )

    # Create tabs for different functionalities
    with gr.Tabs():
        with gr.TabItem(i18n("Translate video")):
            with gr.Row():
                with gr.Column():
                    # Input for video link or file
                    input_video = gr.Textbox(
                        label=i18n("Enter YouTube / BiliBili / Local Video Link"),
                        placeholder=i18n(
                            "Enter a YouTube link, BiliBili link or a local video file path"
                        ),
                        value="",
                    )
                    gr.Markdown(
                        i18n(
                            "You can also select a local video file to translate. Click on the button below."
                        )
                    )
                    input_video_file = gr.File(
                        label=i18n("Select Video File"),
                        file_count="single",
                        type="file",
                    )
                    with gr.Row():
                        # Dropdown for selecting the source language
                        source_language_dropdown = gr.Dropdown(
                            label=i18n("Source language"),
                            choices=LANGUAGE_NAMES,
                            value=SOURCE_LANGUAGE,
                        )
                        # Dropdown for selecting the target language
                        target_language_dropdown = gr.Dropdown(
                            label=i18n("Target language"),
                            choices=LANGUAGE_NAMES,
                            value=TARGET_LANGUAGE,
                        )
                    with gr.Row():
                        # Checkbox for translating from the original audio
                        translate_from_original_audio_checkbox = gr.Checkbox(
                            label=i18n("Translate from the original audio"),
                            value=TRANSLATE_FROM_CLIP,
                        )
                        # Checkbox for translating to the original audio
                        translate_to_original_audio_checkbox = gr.Checkbox(
                            label=i18n("Translate to the original audio"),
                            value=TRANSLATE_TO_CLIP,
                        )
                    with gr.Row():
                        # Dropdown for selecting the TTS speaker
                        tts_speaker_dropdown = gr.Dropdown(
                            label=i18n("TTS speaker"),
                            choices=tts_voices,
                            value=TTS_SPEAKER,
                        )
                        # Dropdown for selecting the TTS language
                        tts_language_dropdown = gr.Dropdown(
                            label=i18n("TTS language"),
                            choices=LANGUAGE_NAMES,
                            value=TTS_LANGUAGE,
                        )
                    with gr.Row():
                        # Slider for adjusting the TTS speed
                        tts_speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            step=0.1,
                            value=TTS_SPEED,
                            label=i18n("TTS speed"),
                            interactive=True,
                        )
                        # Dropdown for selecting the Whisper language
                        whisper_language_dropdown = gr.Dropdown(
                            label=i18n("Whisper language"),
                            choices=LANGUAGE_NAMES,
                            value=WHISPER_LANGUAGE,
                        )
                    with gr.Accordion(i18n("Advanced settings"), open=False):
                        with gr.Row():
                            # Slider for changing the pitch
                            pitch_change_slider = gr.Slider(
                                minimum=-24,
                                maximum=24,
                                step=1,
                                value=PITCH_CHANGE,
                                label=i18n("Pitch change"),
                                interactive=True,
                            )
                            # Slider for setting the index rate
                            index_rate_slider = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=INDEX_RATE,
                                label=i18n("Index rate"),
                                interactive=True,
                            )
                            # Slider for setting the filter radius
                            filter_radius_slider = gr.Slider(
                                minimum=0,
                                maximum=7,
                                step=1,
                                value=FILTER_RADIUS,
                                label=i18n("Filter radius"),
                                interactive=True,
                            )
                        with gr.Row():
                            # Slider for setting the envelope mix
                            envelope_mix_slider = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=ENVELOPE_MIX,
                                label=i18n("Envelope mix"),
                                interactive=True,
                            )
                            # Slider for protecting voiceless consonants
                            protect_voiceless_slider = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                step=0.01,
                                value=PROTECT_VOICELESS,
                                label=i18n("Protect voiceless"),
                                interactive=True,
                            )
                        with gr.Row():
                            # Dropdown for selecting the remix option
                            remix_option_dropdown = gr.Dropdown(
                                label=i18n("Remix option"),
                                choices=["rebalance", "dither"],
                                value=REMIX_OPTION,
                            )
                            #
