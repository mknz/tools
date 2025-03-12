# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faster-whisper",
# ]
# ///
import argparse

from faster_whisper import WhisperModel


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

    # Parse arguments
    args = parser.parse_args()

    # Initialize the Whisper model with user-provided arguments
    model = WhisperModel(args.model_size, compute_type="int8")

    # Transcribe the audio file
    segments, _ = model.transcribe(
        args.audio_file,
        args.language,
        "transcribe",
    )

    # Extract and print the transcribed text
    result = " ".join(segment.text for segment in segments)
    print(result)


if __name__ == "__main__":
    main()
