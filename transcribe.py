# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faster-whisper",
# ]
# ///
import argparse
from datetime import timedelta

from faster_whisper import WhisperModel, BatchedInferencePipeline


def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((td.total_seconds() - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using a Whisper model."
    )

    # Define arguments
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to be transcribed",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code for transcription (default: None [autodetect])",
    )
    parser.add_argument(
        "-m",
        "--model_size",
        choices=["tiny", "small", "base", "medium", "large"],
        default="large",
        help="Model size to use (default: large)",
    )
    parser.add_argument(
        "--multilingual",
        action='store_true',
    )
    parser.add_argument(
        "--initial_prompt",
        default='',
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize the Whisper model with user-provided arguments
    model = WhisperModel(args.model_size, compute_type="int8")

    # Transcribe the audio file
    model_ = model
    segments, _ = model_.transcribe(
        args.audio_file,
        args.language,
        "transcribe",
        multilingual=args.multilingual,
        initial_prompt=args.initial_prompt,
    )

    # Output in srt format
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        print(f"{i}")
        print(f"{start_time} --> {end_time}")
        print(f"{segment.text.strip()}\n")


if __name__ == "__main__":
    main()
