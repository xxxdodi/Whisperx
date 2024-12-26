from typing import List

import whisperx
from huggingface_hub import list_models

import cfg
from model.wraperModel import WraperModel


class SeveralWhispersService:
    listModels: List[WraperModel] = []

    async def getModels(self):
        return self.listModels

    async def initModels(self, listInit):
        for item in listInit:
            asr_options = {
                "multilingual": item.asr_opt_multilingual,  # Использовать мультиязычную модель
                "hotwords": item.asr_opt_hotwords  # Горячие слова, если необходимо
            }
            for _ in range(item.numberOfModelsCPU):
                currentmodel = whisperx.load_model(
                    whisper_arch=item.modelName,
                    device=cfg.cfg["device_cpu"],
                    compute_type=cfg.cfg["compute_type_int8"],
                    asr_options=asr_options
                )
                model = WraperModel(
                    asr_model=currentmodel,
                    device=cfg.cfg["device_cpu"],
                    compute_type=cfg.cfg["compute_type_int8"],
                    modelName=item.modelName,
                    multilingual=item.asr_opt_multilingual,
                    hotwords=item.asr_opt_hotwords,
                )
                self.listModels.append(model)

            for _ in range(item.numberOfModelsGPU):
                currentmodel = whisperx.load_model(
                    whisper_arch=item.modelName,
                    device=cfg.cfg["device_cuda"],
                    compute_type=cfg.cfg["compute_type_f16"],
                    asr_options=asr_options
                )
                model = WraperModel(
                    asr_model=currentmodel,
                    device=cfg.cfg["device_cuda"],
                    compute_type=cfg.cfg["compute_type_f16"],
                    modelName=item.modelName,
                    multilingual=item.asr_opt_multilingual,
                    hotwords=item.asr_opt_hotwords,
                )
                self.listModels.append(model)

        return self.listModels