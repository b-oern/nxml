

from nwebclient import runner as r

class Qwen3TTS(r.BaseJobExecutor):
    """
      from nxml import tts
      q = tts.Qwen3TTS()
      q.generate("Hallo Welt")
    """

    MODULES = ['qwen-tts']

    device = "cuda:0"
    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    language = "German"

    def __init__(self, type='tts'):
        super().__init__(type)
        self.define_vars('device', 'model_name', 'language')

    def generate(self, text, output_file='tts.wav'):
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # single inference
        wavs, sr = model.generate_custom_voice(
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



