from dependency_injector import containers, providers

from service.emotionalRecognitionService import EmotionalRecognitionService


class EmotionalRecognitionContainer(containers.DeclarativeContainer):
    emotionalRecognitionService = providers.Singleton(EmotionalRecognitionService)