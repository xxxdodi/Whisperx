from dataclasses import dataclass
from typing import Optional

from whisperx.asr import FasterWhisperPipeline


@dataclass
class WraperModel:
    asr_model: FasterWhisperPipeline
    device: str
    compute_type: str
    modelName: str
    multilingual: bool
    hotwords: Optional[str] = None
    flagIsBusy: bool = False

