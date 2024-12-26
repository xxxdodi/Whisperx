import datetime
import io
from io import BytesIO

import ffmpeg
import torch
import torchaudio

import cfg
import torch

# Функция для добавления записей в лог
def log(message, level='INFO'):
    #TODO
    # global debug  # Чтобы ссылаться на глобальную переменную
    # if not debug:  # Если debug равно False, выйти из функции
    #     return
    current_time = datetime.datetime.now()
    log_message = f"[{current_time}] [{level}] {message}"
    print(log_message)

def get_device() -> list[str] | str:
    try:
        print("Запуск инициализации CUDA...")
        torch.cuda.init()
        if torch.cuda.is_available():
            log("Инициализация модели с использованием CUDA завершена")
            return [cfg.cfg["device_cuda"], cfg.cfg["compute_type_f16"]]
        else:
            log("Инициализация модели с использованием CPU завершена")
            return [cfg.cfg["device_cpu"], cfg.cfg["compute_type_int8"]]
    except RuntimeError as e:
        log("Не удалось инициализировать модель с CUDA. Используется CPU.")
        return [cfg.cfg["error"], cfg.cfg["error"]]

async def predict_emotion(audio_file, model, feature_extractor, num2emotion):
    waveform, sample_rate = torchaudio.load(uri=audio_file, normalize=True, format="wav")
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = feature_extractor(
        waveform.squeeze(0),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )

    logits = model(inputs.input_values).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = num2emotion[predictions.numpy()[0]]
    return predicted_emotion


async def getEmotion(audioFile, model, feature_extractor, num2emotion):
    audioFile.file.seek(0)
    file_bytes = BytesIO(audioFile.file.read())

    emotion = await predict_emotion(
        audio_file=file_bytes,  # , // Принимает BinaryIO
        model=model,
        feature_extractor=feature_extractor,
        num2emotion=num2emotion
    )

    return emotion

async def getTranscribition(wrappedModel, convertAudio):
    wrappedModel.flagIsBusy = True
    transcrib_result = wrappedModel.asr_model.transcribe(convertAudio, batch_size=16, language="ru")
    wrappedModel.flagIsBusy = False

    return transcrib_result


