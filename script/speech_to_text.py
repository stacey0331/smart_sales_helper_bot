import queue
import sys
import re
import sounddevice as sd
import numpy as np
from google.cloud import speech_v1p1beta1 as speech

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

"""Opens a recording stream as a generator yielding the audio chunks."""
class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.closed = False
        self._stream = sd.InputStream(
            samplerate=self._rate, channels=1, dtype='int16',
            callback=self._callback, blocksize=self._chunk
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time, status):
        """Callback function to receive audio chunks."""
        if status:
            print(f"Error in audio stream callback: {status}")
        self._buff.put(indata.copy())

    def generator(self):
        """Generates audio chunks from the stream."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk.tobytes()

"""Iterates through server responses and prints transcriptions."""
def listen_print_loop(responses):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0

    return transcript