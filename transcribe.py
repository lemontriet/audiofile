#!/usr/bin/env python3
"""
Audio transcription using OpenAI Whisper.

Setup (run once):
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt

Usage:
    python transcribe.py "your_audio_file.mp3"
    python transcribe.py "your_audio_file.mp3" -m small -l en
"""

import argparse
import os
from pathlib import Path


def setup_ffmpeg():
    """Add bundled ffmpeg to PATH before importing whisper."""
    try:
        import imageio_ffmpeg
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
        os.environ["PATH"] = str(ffmpeg_path) + os.pathsep + os.environ["PATH"]
        print(f"Using ffmpeg from: {ffmpeg_path}")
    except ImportError:
        print("Warning: imageio-ffmpeg not installed, assuming ffmpeg is in PATH")


def transcribe(
    audio_path: str,
    model_name: str = "base",
    output_dir: str | None = None,
    language: str | None = None,
) -> str:
    """
    Transcribe an audio file and save the transcript.

    Args:
        audio_path: Path to the audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        output_dir: Directory for output file (defaults to audio file's directory)
        language: Language code (e.g., 'en') or None for auto-detect

    Returns:
        The transcribed text
    """
    # Import whisper after ffmpeg is in PATH
    import whisper

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name)

    print(f"Transcribing {audio_path.name}...")
    result = model.transcribe(str(audio_path), language=language)

    # Determine output path
    output_dir = Path(output_dir) if output_dir else audio_path.parent
    output_path = output_dir / f"{audio_path.stem}_transcript.txt"

    # Write transcript
    output_path.write_text(result["text"].strip(), encoding="utf-8")
    print(f"Saved to {output_path}")

    return result["text"]


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 -m small
    python transcribe.py audio.mp3 -m medium -l en -o ./transcripts
        """,
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: base)",
    )
    parser.add_argument("-o", "--output-dir", help="Output directory")
    parser.add_argument("-l", "--language", help="Language code (e.g., 'en')")

    args = parser.parse_args()

    # Setup ffmpeg before transcribing
    setup_ffmpeg()

    transcribe(args.audio, args.model, args.output_dir, args.language)


if __name__ == "__main__":
    main()