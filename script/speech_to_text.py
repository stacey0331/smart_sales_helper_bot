import queue
import json
import sounddevice as sd
from script.utils import preprocess_text, sentences_to_embeddings

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

avoid_phrase = {"guarantee", "trust me", "just checking in", "to be honest", "what if i said", "mhm", "hm"}

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

def listen_print_loop(responses, open_id, log_reg, glove_model, message_api_client):
    """Iterates through server responses and process transcriptions."""
    for response in responses:
        if not response.results:
            continue

        for result in response.results:
            transcript = result.alternatives[0].transcript
            if len(transcript) == 0:
                continue
            if transcript.lower() in avoid_phrase:
                text_content = {
                    "text": "To-be-avoided phrases detected: " + transcript
                }
                message_api_client.send_text_with_open_id(open_id, json.dumps(text_content))

            if result.is_final:
                # print(f"Transcript: {transcript}")
                
                preprocessed_sentence = preprocess_text(transcript)
                sentence_embedding = sentences_to_embeddings([preprocessed_sentence], glove_model, vector_size=300)
                
                prediction = log_reg.predict(sentence_embedding)[0]
                if prediction == 1: 
                    probability = float(log_reg.predict_proba(sentence_embedding)[0][1])
                    text_content = {
                        "text": "Informal sentence detected: " + transcript + "\n(probability: " + str("{:.2f}".format(probability)) + ")."
                    }
                    message_api_client.send_text_with_open_id(open_id, json.dumps(text_content))
                # print(f"Predicted Result: {prediction}") # 1 means informal
                # print(f"Informal Probability: {probability}")