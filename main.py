import copy
import io
import sys
from io import BytesIO
from sys import modules
from typing import List, BinaryIO

from anyio import sleep
from fastapi import FastAPI, UploadFile, Depends, File, HTTPException, Request, Body, Form
from dependency_injector.wiring import inject, Provide
from typing import Optional

from sqlalchemy.sql.functions import random

import cfg
from container.emotionalRecognitionContainer import EmotionalRecognitionContainer
from container.mainContainer import Container
from container.severalWhispersContainer import SeveralWhispersContainer
from container.whisperContainer import WhisperContainer
from container.convertAudioContainer import ConvertAudioContainer
from model.whisperRequestModel import WhisperRequestModel
from service.emotionalRecognitionService import EmotionalRecognitionService

from service.severalWhispersService import SeveralWhispersService
from service.whisperService import WhisperService
from service.convertAudioService import ConvertAudioService

from utilits.utilits import log, get_device, predict_emotion, getEmotion, getTranscribition
import whisperx
import logging

logger = logging.getLogger('uvicorn.error')
app = FastAPI()


@app.api_route('/')
@inject
async def index(service: Service = Depends(Provide[Container.service])):
    result = await service.process()
    return {'result': result}


@app.on_event("startup")
@inject
async def startup_event(
        service: Service = Depends(Provide[Container.service]),
        whisperService: WhisperService = Depends(Provide[WhisperContainer.whisperService]),
        emotionalRecognitionService: EmotionalRecognitionService = Depends(Provide[EmotionalRecognitionContainer.emotionalRecognitionService]),
):
    conf = get_device()
    await whisperService.initModel(asr_model=cfg.cfg["model"], device=conf[0], compute_type=conf[1])
    # await whisperService.initModel(asr_model=cfg.cfg["model"], device="cpu", compute_type="int8")
    await emotionalRecognitionService.initModels()

@app.api_route('/asr', methods=['POST'])
@inject
async def asr_endpoint(
        audioFile: UploadFile = File(...),
        whisperService: WhisperService = Depends(Provide[WhisperContainer.whisperService]),
        convertAudioService: ConvertAudioService = Depends(Provide[ConvertAudioContainer.convertAudioService]),
        emotionalRecognitionService: EmotionalRecognitionService = Depends(
            Provide[EmotionalRecognitionContainer.emotionalRecognitionService]),
):
    model = await whisperService.modelReturn()
    log("Incomming to /asr endpoint")
    try:
        convertAudio = await convertAudioService.convertAudio(audioFile.file)
        transcrib_result = model.transcribe(convertAudio, batch_size=16, language="ru")

        audioFile.file.seek(0)
        file_bytes = BytesIO(audioFile.file.read())

        emotion = await predict_emotion(
            audio_file = file_bytes, #, // Принимает BinaryIO
            model=emotionalRecognitionService.model,
            feature_extractor=emotionalRecognitionService.feature_extractor,
            num2emotion=emotionalRecognitionService.num2emotion
        )
        return {"message": f"выполнен на дефолтном whisperx сервисе",
                "transcribe": f"{transcrib_result}",
                "emotion": f"{emotion}"
                }
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail='Something went wrong')
    return {"message": f"Successfully uploaded {audioFile.filename}"}


@app.api_route('/initWhispers', methods=['POST'])
@inject
async def initWhispers_endpoint(
        request: Request,
        severalWhispersService: SeveralWhispersService = Depends(
            Provide[SeveralWhispersContainer.severalWhispersService])

):
    raw_data = await request.json()

    # Десериализуем JSON в список объектов модели
    listInit: List[WhisperRequestModel] = [WhisperRequestModel(**item) for item in raw_data]
    listOfModels = await severalWhispersService.initModels(listInit)
    return listInit


@app.api_route('/sendAudio', methods=['POST'])
@inject
async def sendAudio_endpoint(
        severalWhispersService: SeveralWhispersService = Depends(
            Provide[SeveralWhispersContainer.severalWhispersService]),
        convertAudioService: ConvertAudioService = Depends(Provide[ConvertAudioContainer.convertAudioService]),
        emotionalRecognitionService: EmotionalRecognitionService = Depends(
            Provide[EmotionalRecognitionContainer.emotionalRecognitionService]),
        audioFile: UploadFile = File(...),
        preferredModelName: Optional[str] = Form(None),
        device: Optional[str] = Form(None)
):
    listOfModels = await severalWhispersService.getModels()
    convertAudio = await convertAudioService.convertAudio(audioFile.file)

    #Есть ли инициализированные whispex сервисы
    if not listOfModels:
        return {"message": f"Нет инициализированных моделей в пуле. Выполните инициализацию /initWhispers"}

    # Если нет указаний на какой модели запускать, ставим в очаредь на первый
    if not preferredModelName:
        firstWhisperService = listOfModels[0]

        transcrib_result = await getTranscribition(
            wrappedModel=firstWhisperService,
            convertAudio=convertAudio
        )

        emotion = await getEmotion(
            audioFile=audioFile,  # , // Принимает BinaryIO
            model=emotionalRecognitionService.model,
            feature_extractor=emotionalRecognitionService.feature_extractor,
            num2emotion=emotionalRecognitionService.num2emotion
        )

        return {"message": f'Выполнен на случайном {firstWhisperService.modelName} сервисе',
                "result": f'{transcrib_result}',
                "emotion": f'{emotion}',
                "device": f'{firstWhisperService.device}'
                }


    # Ищем в списке нужный нам сервис whisperx и на нем запускаем
    for item in listOfModels:
        if item.modelName == preferredModelName:
            if item.device == device:
                if not item.flagIsBusy:
                    transcrib_result = await getTranscribition(
                        wrappedModel=item,
                        convertAudio=convertAudio
                    )

                    emotion = await getEmotion(
                        audioFile=audioFile,  # , // Принимает BinaryIO
                        model=emotionalRecognitionService.model,
                        feature_extractor=emotionalRecognitionService.feature_extractor,
                        num2emotion=emotionalRecognitionService.num2emotion
                    )

                    return {"message": f'Выполнен на целевом {item.modelName} сервисе',
                            "result": f'{transcrib_result}',
                            "emotion": f'{emotion}',
                            "device": f'{item.device}'
                            }

    # Если подходящий сервис не найден, пытайемся запустить на перовой свободной
    for item in listOfModels:
        if not item.flagIsBusy:
            transcrib_result = await getTranscribition(
                wrappedModel=item,
                convertAudio=convertAudio
            )

            emotion = await getEmotion(
                audioFile=audioFile,  # , // Принимает BinaryIO
                model=emotionalRecognitionService.model,
                feature_extractor=emotionalRecognitionService.feature_extractor,
                num2emotion=emotionalRecognitionService.num2emotion
            )
            return {"message": f'Выполнен на первом свободном {item.modelName} сервисе',
                    "result": f'{transcrib_result}',
                    "emotion": f'{emotion}',
                    "device": f'{item.device}'
                    }

    # Если все заняты, то ставим в очаредь на случайную
    firstWhisperService = listOfModels[0]

    transcrib_result = await getTranscribition(
        wrappedModel=firstWhisperService,
        convertAudio=convertAudio
    )

    emotion = await getEmotion(
        audioFile=audioFile,  # , // Принимает BinaryIO
        model=emotionalRecognitionService.model,
        feature_extractor=emotionalRecognitionService.feature_extractor,
        num2emotion=emotionalRecognitionService.num2emotion
    )

    return {"message": f'Выполнен на случайном {firstWhisperService.modelName} сервисе',
            "result": f'{transcrib_result}',
            "emotion": f'{emotion}',
            "device": f'{item.device}'
            }



whisperContainer = WhisperContainer()
audioContainer = ConvertAudioContainer()
severalWhispersContainer = SeveralWhispersContainer()
emotionalRecognitionContainer = EmotionalRecognitionContainer()


whisperContainer.wire(modules=[sys.modules[__name__]])
audioContainer.wire(modules=[sys.modules[__name__]])
severalWhispersContainer.wire(modules=[sys.modules[__name__]])
emotionalRecognitionContainer.wire(modules=[sys.modules[__name__]])
