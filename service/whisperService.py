import whisperx

class WhisperService:

    model=None
    default_asr_options = {
        "multilingual": True,  # Использовать мультиязычную модель
        "hotwords": None  # Горячие слова, если необходимо
    }

    async def initModel(self,asr_model,device,compute_type):
        self.model = whisperx.load_model(whisper_arch=asr_model,device=device,compute_type=compute_type,asr_options=self.default_asr_options)

    async def modelReturn(self)->any:
        return self.model

    async def process(self) -> str:
        return 'Ok'