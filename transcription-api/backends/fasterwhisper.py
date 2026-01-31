import numpy as np
from .backend import Backend, Transcription, Segment
import os, math
from tqdm import tqdm  # type: ignore
import uuid
from faster_whisper import WhisperModel, download_model, decode_audio
import torch
import logging

class FasterWhisperBackend(Backend):
    device: str = "cpu"  # cpu, cuda
    quantization: str = "int8"  # int8, float16
    model: WhisperModel | None = None

    def __init__(self, model_size, device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.__post_init__()

    def model_path(self) -> str:
        local_model_path = os.path.join(
            os.environ["WHISPER_MODELS_DIR"], f"faster-whisper-{self.model_size}"
        )

        if os.path.exists(local_model_path):
            return local_model_path
        else:
            raise RuntimeError(f"model not found in {local_model_path}")


    def load(self) -> None:
        cpu_threads = int(os.environ.get("CPU_THREADS", 4))

        # Device: per defecte 'cuda' si existeix GPU visible, sinó 'cpu'
        device = os.environ.get("WHISPER_DEVICE", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        # Quantització: float16 quan cuda, int8 quan cpu (pots ajustar-ho)
        quant = os.environ.get("WHISPER_COMPUTE_TYPE",
                           getattr(self, "quantization", "float16" if device == "cuda" else "int8"))

        self.device = device
        self.quantization = quant

        self.model = WhisperModel(
            self.model_path(),
            device=self.device,
            compute_type=self.quantization,
            cpu_threads=cpu_threads,
        )

        logging.warning(
            "WhisperModel initialized: device=%s compute_type=%s cpu_threads=%s",
            self.device, self.quantization, cpu_threads
        )
        
    def get_model(self) -> None:
        print(f"Downloading model {self.model_size}...")
        local_model_path = os.path.join(os.environ["WHISPER_MODELS_DIR"], f"faster-whisper-{self.model_size}")
        local_model_cache = os.path.join(os.environ["WHISPER_MODELS_DIR"], f"faster-whisper-{self.model_size}", "cache")
        # Check if directory exists
        if not os.path.exists(local_model_path):
            os.makedirs(local_model_path)
        try:
            download_model(self.model_size, output_dir=local_model_path, local_files_only=True, cache_dir=local_model_cache)
            print("Model already cached...")
        except:
            download_model(self.model_size, output_dir=local_model_path, local_files_only=False, cache_dir=local_model_cache)

    def transcribe(
        self,
        input: np.ndarray,
        silent: bool = False,
        language: str = None,
        beam_size: int = 5,
        initial_prompt: str = None,
        hotwords: list[str] = None,
    ) -> Transcription:
        """
        Return word level transcription data.
        World level probabilities are calculated by ctranslate2.models.Whisper.align
        Accepts additional parameters: beam_size, initial_prompt, hotwords.
        """
        result: list[Segment] = []
        assert self.model is not None
        segments, info = self.model.transcribe(
            input,
            beam_size=beam_size,
            word_timestamps=True,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        # ps = playback seconds
        with tqdm(
            total=info.duration, unit_scale=True, unit="ps", disable=silent
        ) as pbar:
            for segment in segments:
                if segment.words is None:
                    continue
                id = uuid.uuid4().hex
                segment_extract: Segment = {
                    "id": id,
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "score": round(math.exp(segment.avg_logprob), 2),
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                            "score": round(w.probability, 2),
                        }
                        for w in segment.words
                    ],
                }
                result.append(segment_extract)
                if not silent:
                    pbar.update(segment.end - pbar.last_print_n)
        text = " ".join([segment["text"] for segment in result])
        text = ' '.join(text.strip().split())
        transcription: Transcription = {
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "segments": result,
        }
        return transcription
