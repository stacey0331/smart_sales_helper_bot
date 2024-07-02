import queue
import sounddevice as sd

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
    for response in responses:
        if not response.results:
            continue

        for result in response.results:
            if result.is_final:
                transcript = result.alternatives[0].transcript
                print(f"Transcript: {transcript}")

    return transcript