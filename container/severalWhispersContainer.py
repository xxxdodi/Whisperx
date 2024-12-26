from dependency_injector import containers, providers
from service.severalWhispersService import SeveralWhispersService

class SeveralWhispersContainer(containers.DeclarativeContainer):
    severalWhispersService = providers.Singleton(SeveralWhispersService)
