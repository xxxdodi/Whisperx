from dependency_injector import containers, providers
from service.convertAudioService  import ConvertAudioService

class ConvertAudioContainer(containers.DeclarativeContainer):
    convertAudioService = providers.Factory(ConvertAudioService)
