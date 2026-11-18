

from nwebclient import runner as r
from nwebclient import dev as d

from nxml.hugging import HuggingRunner


class Qwen3TTS(HuggingRunner, r.BaseJobExecutor):
    """
      Text-to-Speech mit Qwen3-TTS (~4 GB VRAM, 1,7B Parameter bf16).

      from nxml import tts
      q = tts.Qwen3TTS()
      q.generate("Hallo Welt")

    Das Modell wird lazy beim ersten Aufruf geladen und bleibt geladen.
    """
    TAGS = ['tts', 'hugging']
    MODULES = ['qwen-tts', 'soundfile', 'torch']

    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    language = "German"
    HF_MODEL_ATTRS = ('model',)

    def __init__(self, type='tts', model_name=None, device_mode='auto'):
        super().__init__(type)
        self.hf_init(model_name, device_mode)
        self.define_vars('language')
        self.define_sig(d.PStr('op', 'tts'), d.Param('text'), d.PStr('filename', 'tts.wav'))\
            .set_description("Erzeugt Sprache aus Text und speichert sie als WAV-Datei").r_success()

    def load_model(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=torch.float32 if self.device == 'cpu' else torch.bfloat16,
            # flash-attn gibt es nur für CUDA
            attn_implementation="flash_attention_2" if self.device == 'cuda' else 'sdpa',
        )

    def generate(self, text, output_file='tts.wav'):
        import soundfile as sf
        self.load()
        # single inference
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=self.language,
            # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
            speaker="Vivian",
            instruct="neutral",  # Omit if not needed.
        )
        sf.write(output_file, wavs[0], sr)

    def execute_tts(self, data):
        self.generate(data['text'], data.get('filename', 'tts.wav'))
        return self.success()
