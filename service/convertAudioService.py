import io
import numpy as np
import ffmpeg
from io import BytesIO

class ConvertAudioService:
    async def convertAudio(self, audioFile) -> any:
        # Обработка через FFmpeg
        input_stream = BytesIO(audioFile.read())

        output_stream = (
            ffmpeg
            .input("pipe:0")  # Вход через stdin
            .output("pipe:1", format="s16le", acodec="pcm_s16le", ar=16000)  # Вывод через stdout
            .run(input=input_stream.read(), capture_stdout=True, capture_stderr=True)
        )
        stdout_data = output_stream[0]
        byte_object_audio = io.BytesIO(stdout_data).read()
        numpy_array_audio= np.frombuffer(byte_object_audio, np.int16).flatten().astype(np.float32) / 32768.0

        return numpy_array_audio
