from dataclasses import dataclass

@dataclass
class WhisperRequestModel:
    modelName: str
    numberOfModelsCPU: int
    numberOfModelsGPU: int
    asr_opt_multilingual: bool
    asr_opt_hotwords: bool
