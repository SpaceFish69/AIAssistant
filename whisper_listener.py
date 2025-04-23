import sounddevice as sd
import numpy as np
import whisper
import tempfile
import threading
import queue
import time

model = whisper.load_model("base")
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def transcribe_live_audio(callback):
    def stream_audio():
        with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
            while True:
                if not q.empty():
                    chunk = q.get()
                    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                        np.save(f, chunk)
                        f.flush()
                        audio_data = np.fromfile(f.name, dtype=np.float32)
                        result = model.transcribe(audio_data)
                        callback(result["text"])
    threading.Thread(target=stream_audio, daemon=True).start()