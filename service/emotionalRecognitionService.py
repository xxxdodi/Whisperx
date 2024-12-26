from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification


class EmotionalRecognitionService:

    feature_extractor = None
    model = None
    num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}

    async def initModels(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned")

    async def getEmotionalCridentials(self):
        return [self.feature_extractor, self.model, self.num2emotion]
