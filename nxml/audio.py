import base64
import json
import hashlib
import time

from nwebclient import runner as r, base
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import dev as d

from nxml.hugging import HuggingRunner


class LibRosa(r.BaseJobExecutor):
    """
    Analysiert Audiodateien mit librosa, z.B. Tempo und Mel-Spektrogramm.
    """
    MODULES = ['librosa']
    TAGS = ['audio-in']

    def __init__(self):
        super().__init__('librosa')
        self.define_sig(d.Param('file')).set_description("Schätzt das Tempo (BPM) einer Audiodatei").r(tempo='float')

    def execute_file(self, file):
        import librosa
        import librosa.feature
        sr = 44100
        y, sr = librosa.load(file)
        tempo = librosa.feature.tempo(y=y, sr=sr)
        return {
            'tempo': tempo[0]
        }

    def mel(self, file):
        import numpy as np
        import librosa
        import librosa.feature
        y, sr = librosa.load(file)
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(librosa.power_to_db(s, ref=np.max),
                                       x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        img = librosa.display.specshow(s, x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title='MFCC')
        import io
        buffer = io.BytesIO()
        fig.savefig(buffer, format='jpg')
        buffer.seek(0)
        return 'data:image/jpg;base64' + base64.b64encode(buffer.read()).decode()

    def execute(self, data):
        if 'file' in data:
            return self.execute_file(data['file'])
        return super().execute(data)


class CommandRecognition(HuggingRunner, r.BaseJobExecutor):
    """
      Erkennt ein Aktivierungswort in einem Audio-Stream (~0,4 GB VRAM, AST mit 87M Parametern fp32).
    """
    TAGS = ['hugging']
    MODULES = ['datasets', 'soundfile', 'librosa', 'transformers']

    model_name = "MIT/ast-finetuned-speech-commands-v2"
    HF_MODEL_ATTRS = ('classifier',)

    def __init__(self, model=None, device_mode='auto'):
        super().__init__('speech_reg')
        self.hf_init(model, device_mode)
        self.define_sig(d.Param('audio')).set_description("Erkennt gesprochene Befehlswörter in Audio")\
            .r(result='list')
        self.param_names['audio'] = "{'path': 'down/1816b768_nohash_0.wav', 'array': array([ 0.00079346, -0.00543213, -0.00054932, ..., -0.00717163,-0.00415039, -0.00811768]), 'sampling_rate': 16000}"

    def load_model(self):
        self.classifier = self.hf_pipeline("audio-classification")

    def predict(self, audio):
        self.load()
        return {'result': self.classifier(audio)}
        # [{'score': 0.9999892711639404, 'label': 'backward'},{'score': 1.7504888774055871e-06, 'label': 'happy'}, ...

    def execute(self, data):
        if 'audio' in data:
            return self.predict(data['audio'])
        return super().execute(data)


class VoiceAssitant(HuggingRunner, r.BaseJobExecutor):
    """
    Wartet am Mikrofon auf ein Aktivierungswort (~0,4 GB VRAM, AST mit 87M Parametern fp32).

    # https://huggingface.co/learn/audio-course/chapter4/classification_models
    # https://huggingface.co/learn/audio-course/chapter7/voice-assistant

    ffmpeg

    via https://huggingface.co/learn/audio-course/chapter7/voice-assistant
    """
    TAGS = ['hugging']
    MODULES = ['transformers', 'torch']

    model_name = "MIT/ast-finetuned-speech-commands-v2"
    HF_MODEL_ATTRS = ('classifier',)

    def __init__(self, model=None, device_mode='auto'):
        super().__init__()
        self.hf_init(model, device_mode)

    def load_model(self):
        self.classifier = self.hf_pipeline("audio-classification")

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
        self.load()
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


class AudioGenerator(HuggingRunner, r.BaseJobExecutor):
    """
    Erzeugt Musik aus einer Textbeschreibung (MusicGen) und speichert sie als WAV-Datei (~2,5 GB VRAM, musicgen-small fp32 inkl. T5-Encoder).

    see also https://huggingface.co/spaces/artificialguybr/Stable-Audio-Open-Zero
    https://huggingface.co/facebook/musicgen-small

      g = AudioGenerator()
      g.execute({'op': 'generate', 'prompt': 'lo-fi music with a soothing melody', 'filename': 'lofi.wav'})

    Das Modell wird lazy beim ersten Aufruf geladen.
    """
    TAGS = ['audio-out', 'hugging']
    MODULES = ['transformers', 'torch', 'scipy', 'numpy']

    model_name = "facebook/musicgen-small"
    # MusicGen erzeugt 50 Audio-Tokens pro Sekunde, 256 Tokens sind ca. 5 s
    max_new_tokens = 256

    HF_MODEL_ATTRS = ('synthesiser',)

    def __init__(self, type='audio_gen', model_name=None, device_mode='auto'):
        """
            :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        """
        super().__init__(type)
        self.hf_init(model_name, device_mode)
        self.define_vars('max_new_tokens')
        self.define_sig(d.PStr('op', 'generate'), d.Param('prompt'), d.PStr('filename', 'musicgen_out.wav'),
                        d.PInt('max_new_tokens', self.max_new_tokens))\
            .set_description("Erzeugt Musik aus einer Textbeschreibung und speichert sie als WAV-Datei")\
            .r_success(filename='str', sampling_rate='int', duration='float')

    def load_model(self):
        self.synthesiser = self.hf_pipeline("text-to-audio")

    def generate(self, prompt="lo-fi music with a soothing melody", filename='musicgen_out.wav', max_new_tokens=None):
        """
            :return: {'filename': str, 'sampling_rate': int, 'duration': float}
        """
        import numpy as np
        from scipy.io import wavfile
        self.load()
        music = self.synthesiser(prompt, forward_params={
            'do_sample': True,
            'max_new_tokens': int(max_new_tokens or self.max_new_tokens)
        })
        rate = int(music['sampling_rate'])
        # Pipeline liefert (Kanäle, Samples), scipy erwartet (Samples, Kanäle) bzw. 1D bei Mono
        audio = np.asarray(music['audio'], dtype=np.float32)
        while audio.ndim > 2:
            audio = audio[0]
        if audio.ndim == 2:
            audio = audio[0] if audio.shape[0] == 1 else audio.T
        wavfile.write(filename, rate=rate, data=audio)
        return {'filename': filename, 'sampling_rate': rate, 'duration': round(len(audio) / rate, 3)}

    def execute_generate(self, data):
        if not data.get('prompt'):
            return self.fail("prompt fehlt")
        res = self.generate(data['prompt'], data.get('filename') or 'musicgen_out.wav', data.get('max_new_tokens'))
        return self.success(**res)

    def execute(self, data):
        if 'prompt' in data and 'op' not in data:
            return self.execute_generate(data)
        return super().execute(data)


class FFmpeg(r.BaseJobExecutor):
    """
    Liest mit ffmpeg Informationen über Mediendateien aus, aktuell die Dauer.
    """
    MODULES = ['ffmpeg-python']
    
    def __init__(self):
        super().__init__('ffmpeg')
        self.define_sig(d.Param('duration')).set_description("Dauer einer Mediendatei (Pfad) in Sekunden")\
            .r(duration='str')
    
    def execute(self, data):
        import ffmpeg
        if 'duration' in data:
            return {'duration': ffmpeg.probe(data['duration'])['format']['duration']}
        # print(ffmpeg.probe('in.mp4')['format']['duration'])
        return super().execute(data)


class PiperTTS(r.BaseJobExecutor):
    """
     TTS für die deutsche Sprache

    https://www.thorsten-voice.de/thorsten-voice-%f0%9f%92%9b-piper/
    https://github.com/rhasspy/piper

    Path to piper-bin in args:pipertts

    /home/pi/tts/piper
    """
    TAGS = ['tts']
    MODULES = ['piper-tts', 'Flask']
    CFGS = ['pipertts']

    models = {
        'de': {
            'de_DE-thorsten-high.onnx': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true',
            'de_DE-thorsten-high.onnx.json': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json?download=true.json'
        }
    }

    def __init__(self, type='pipertts', path=None, args: u.Args = {}):
        super().__init__(type)
        self.define_sig(d.Param('text', is_pos=False)).set_description("Erzeugt Sprache (WAV) aus Text")\
            .r_success(file='str', cmd='str')
        self.define_sig(d.PStr('output', '1')).set_description("Liefert die zuletzt erzeugte WAV-Datei")
        self.define_vars('path', 'file')
        self.file = None
        if path is None:
            path = args.get('pipertts', '')
        self.path = path
        self.delayed(13, lambda: u.download_resources(self.path, self.models['de']))

    def tts(self, text):
        self.file = u.hash(text) + '.wav'
        cmd = f'echo "{text}" | {self.path}piper -m {self.path}de_DE-thorsten-high.onnx -f {self.path}{self.file}'
        p = r.ProcessExecutor(cmd)  # on_up=lambda proc: proc.stdin.write(text+"\n")
        p.waitForEnd()
        return self.success(file=self.path+self.file, cmd=cmd)

    def execute(self, data):
        if 'text' in data:
            return self.tts(data['text'])
        elif 'output' in data:
            from flask import send_file
            return send_file(self.path + self.file)
        return super().execute(data)

    def part_index(self, p: base.Page, params={}):
        with p.section(h="Piper TTS"):
            audio_url = f'../../?type={self.type}&output=1'
            with p.t('div', id='pa1'):
                p.audio(audio_url, id='a1')
            p.form_input('text', "Text", id='text')
            jobdef = dict(title='TTS', type=self.type, text='#text')
            p.script("""
                const piper_url = '""" + audio_url + """';
                function on_pipertts(data) {
                    document.getElementById('a1').remove();
                    const a = document.createElement('audio');
                    a.id = 'a1';
                    a.src = piper_url;
                    a.controls = 'controls';
                    document.getElementById('pa1').append(a);
                }
            """)
            p(self.action_btn_parametric("TTS", jobdef, 'on_pipertts'))
        with p.section(h="Debug"):
            p.pre('', id='result')
            p.hr()
            p.prop("Path", self.path)
            p.pre("TODO: Delete old wavs")


class ElevenLabs(r.BaseJobExecutor):
    """
      "tts": "nxml.audio:ElevenLabs"
    """
    TAGS = ['tts', 'external-api']
    type = 'tts'
    CHUNK_SIZE =1024
    voices = {
        'Otto': 'FTNCalFNG5bRnkkaP5Ug'
    }
    def __init__(self, api_key=None, voice_id='FTNCalFNG5bRnkkaP5Ug', args:u.Args={}):
        super().__init__()
        if api_key is None:
            api_key = args.get("elevenlabs_api_key", '')
        self.api_key = api_key
        self.voice_id = voice_id
        self.define_sig(d.Param('tts')).set_description("Erzeugt Sprache (MP3) aus Text über ElevenLabs")\
            .r(filename='str', data='str', response='int')

    def url(self):
        return 'https://api.elevenlabs.io/v1/text-to-speech/' + self.voice_id

    def request(self, text: str):
        import requests
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        filename = str(hashlib.md5(text.encode()).hexdigest()) + '.mp3'
        response = requests.post(self.url(), json=data, headers=headers, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"[util.download] Faild, Status: {response.status_code}")
        return {
            'filename': filename,
            'data': base64.b64encode(u.file_get_contents(filename)).decode('utf-8'),
            'response': response.status_code
        }

    def user_subscription(self):
        url = 'https://api.elevenlabs.io/v1/user/subscription'
        # {
        #   "tier": "<string>",
        #   "character_count": 123,
        #   "character_limit": 123,
        #   "can_extend_character_limit": true,
        #   "allowed_to_extend_character_limit": true,
        #   "next_character_count_reset_unix": 123,
        #   "voice_limit": 123,
        #   "max_voice_add_edits": 123,
        #   "voice_add_edit_counter": 123,
        #   "professional_voice_limit": 123,
        #   "can_extend_voice_limit": true,
        #   "can_use_instant_voice_cloning": true,
        #   "can_use_professional_voice_cloning": true,
        #   "currency": "usd",
        #   "status": "trialing",
        #   "billing_period": "monthly_period",
        #   "character_refresh_period": "monthly_period",
        #   "next_invoice": {
        #     "amount_due_cents": 123,
        #     "next_payment_attempt_unix": 123
        #   },
        #   "has_open_invoices": true
        # }
        return {}

    def execute(self, data):
        if 'tts' in data:
            return self.request(data['tts'])
        return super().execute(data)

    def part_index(self, p: b.Page, params={}):
        p.h1("TTS - Elevenlabs")

