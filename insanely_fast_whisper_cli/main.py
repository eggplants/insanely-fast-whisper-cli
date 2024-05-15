from __future__ import annotations

import time
from pathlib import Path

import click
import torch
from transformers import WhisperForConditionalGeneration, pipeline

__TOKENIZER_BY_LANG: dict[str, str] = {
    "ja": "cl-tohoku/bert-base-japanese",
    "en": "bert-base-uncased",
}


@click.command()  # type: ignore[misc]
@click.argument("audio_file", type=str)  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--model",
    default="openai/whisper-base",
    help=" ".join(  # noqa: FLY002
        [
            "ASR model to use for speech recognition.",
            'Default is "openai/whisper-base".',
            "Model sizes include base, small, medium, large, large-v2.",
            'Additionally, try appending ".en" to model names for English-only applications (not available for large).',
        ],
    ),
)
@click.option(  # type: ignore[misc]
    "--device",
    default="cuda:0",
    help='Device to use for computation. Default is "cuda:0". If you want to use CPU, specify "cpu".',
)
@click.option(  # type: ignore[misc]
    "--dtype",
    default="float32",
    help='Data type for computation. Can be either "float32" or "float16". Default is "float32".',
)
@click.option(  # type: ignore[misc]
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for processing. This is the number of audio files processed at once. Default is 8.",
)
@click.option(  # type: ignore[misc]
    "--chunk-length",
    type=int,
    default=30,
    help="Length of audio chunks to process at once, in seconds. Default is 30 seconds.",
)
@click.option(  # type: ignore[misc]
    "--language",
    type=click.Choice(__TOKENIZER_BY_LANG.keys()),
    help="Language of audio.",
)
@click.option(  # type: ignore[misc]
    "--better-transformer",
    is_flag=True,
    help="Flag to use BetterTransformer for processing. If set, BetterTransformer will be used.",
)
def main(  # noqa: PLR0913
    model: str,
    device: str,
    dtype: str,
    batch_size: int,
    chunk_length: int,
    language: str | None,
    audio_file: str,
    *,
    better_transformer: bool,
) -> None:
    pretrained_model = None
    if language:
        pretrained_model = WhisperForConditionalGeneration.from_pretrained(model)
        pretrained_model.generation_config.language = language

    # Initialize the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=pretrained_model or model,
        device=device,
        torch_dtype=torch.float16 if dtype == "float16" else torch.float32,
        **{"tokenizer": __TOKENIZER_BY_LANG[language]} if language in __TOKENIZER_BY_LANG else {},
    )

    if better_transformer:
        pipe.model = pipe.model.to_bettertransformer()

    # Perform ASR
    click.echo("[+] Model loaded.")
    start_time = time.perf_counter()
    outputs = pipe(
        audio_file,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        return_timestamps=True,
    )

    # Output the results
    click.echo("[+] Transcription completed.")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    click.echo(f"[+] ASR took {elapsed_time:.2f} seconds.")

    # Save ASR chunks to an SRT file
    audio_file_name = Path(audio_file).stem
    srt_filename = Path(f"{audio_file_name}.srt")
    with Path.open(srt_filename, "w") as srt_file:
        prev = 0
        for idx, chunk in enumerate(outputs["chunks"]):
            prev, start_timecode = __seconds_to_srt_time_format(prev, chunk["timestamp"][0])
            prev, end_timecode = __seconds_to_srt_time_format(prev, chunk["timestamp"][1])
            srt_file.write(f"{idx + 1}\n")
            srt_file.write(f"{start_timecode} --> {end_timecode}\n")
            srt_file.write(f"{chunk['text'].strip()}\n\n")
    click.echo(f"[+] Saved: {srt_filename}")


def __seconds_to_srt_time_format(prev: int, seconds: float) -> tuple[int, str]:
    if not (isinstance(seconds, (float, int))):
        seconds = prev
    else:
        prev = seconds  # type: ignore[assignment]
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return (prev, f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}")


__all__ = ("main",)
