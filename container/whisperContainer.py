from dependency_injector import containers, providers
from service.whisperService import WhisperService

class WhisperContainer(containers.DeclarativeContainer):
    whisperService = providers.Singleton(WhisperService)
