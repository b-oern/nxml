

from nwebclient import runner as r


class LibRosa(r.BaseJobExecutor):
    MODULES = ['librosa']
    type = 'librosa'

    def execute_file(self, file):
        import librosa
        import librosa.feature
        sr = 44100
        y, sr = librosa.load(file)
        tempo = librosa.feature.tempo(y=y, sr=sr)
        return {
            'tempo': tempo[0]
        }

    def execute(self, data):
        if 'file' in data:
            return self.execute_file(data['file'])
        return super().execute(data)

class CommandRecognition(r.BaseJobExecutor):
    """

    pip install git+https://github.com/huggingface/transformers
    """
    MODULES = ['datasets, soundfile, librosa']

    type = 'speech_reg'

    def __init__(self, model="MIT/ast-finetuned-speech-commands-v2"):
        from transformers import pipeline
        self.classifier = pipeline("audio-classification", model=model)
        self.param_names['audio'] = "{'path': 'down/1816b768_nohash_0.wav', 'array': array([ 0.00079346, -0.00543213, -0.00054932, ..., -0.00717163,-0.00415039, -0.00811768]), 'sampling_rate': 16000}"

    def predict(self, audio):
        return {'result': self.classifier(audio)}
        # [{'score': 0.9999892711639404, 'label': 'backward'},{'score': 1.7504888774055871e-06, 'label': 'happy'}, ...

    def execute(self, data):
        if 'audio' in data:
            return self.predict(data['audio'])
        return super().execute(data)

# https://huggingface.co/learn/audio-course/chapter4/classification_models
# https://huggingface.co/learn/audio-course/chapter7/voice-assistant

class VoiceAssitant(r.BaseJobExecutor):
    """
    ffmpeg

    via https://huggingface.co/learn/audio-course/chapter7/voice-assistant
    """
    def __init__(self, model="MIT/ast-finetuned-speech-commands-v2"):
        from transformers import pipeline
        self.classifier = pipeline("audio-classification", model=model)

    def launch_fn(self, wake_word="marvin", prob_threshold=0.5, chunk_length_s=2.0, stream_chunk_s=0.25, debug=False):
        """

        dict_keys(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
            'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
            'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])

        :param wake_word:
        :param prob_threshold:
        :param chunk_length_s:
        :param stream_chunk_s:
        :param debug:
        :return:
        """
        from transformers.pipelines.audio_utils import ffmpeg_microphone_live
        if wake_word not in self.classifier.model.config.label2id.keys():
            raise ValueError(f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {self.classifier.model.config.label2id.keys()}.")
        sampling_rate = self.classifier.feature_extractor.sampling_rate
        mic = ffmpeg_microphone_live(sampling_rate=sampling_rate, chunk_length_s=chunk_length_s,   stream_chunk_s=stream_chunk_s)
        print("Listening for wake word...")
        for prediction in self.classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threshold:
                    return True

    def execute(self, data):
        return super().execute(data)
