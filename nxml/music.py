from nwebclient import runner as r
from nwebclient import dev as d

from nxml.hugging import HuggingRunner


class HeartMuLa(HuggingRunner, r.BaseJobExecutor):
    """
      Musikgenerierung aus Songtext und Tags mit HeartMuLa (https://github.com/HeartMuLa/heartlib).
      HeartMuLa-oss-3B (bf16) + HeartCodec (fp32), ca. 16+ GB VRAM, mit lazy_load weniger.

      from nxml import music
      m = music.HeartMuLa()
      m.generate("[Verse]\\nThe sun creeps in across the floor", "piano,happy,romantic", "song.mp3")

    Tags sind kommagetrennt ohne Leerzeichen. Lyrics/Tags dürfen auch Dateipfade sein.
    Die Modelle werden in den Hugging-Face-Cache geladen (lazy beim ersten Aufruf).
    Installation: pip install git+https://github.com/HeartMuLa/heartlib.git
    """
    TAGS = ['audio-out', 'music']
    MODULES = ['git+https://github.com/HeartMuLa/heartlib.git', 'torch', 'torchaudio', 'huggingface_hub']

    # Repo mit tokenizer.json und gen_config.json
    model_name = "HeartMuLa/HeartMuLaGen"
    mula_model = "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"
    codec_model = "HeartMuLa/HeartCodec-oss-20260123"
    HF_MODEL_ATTRS = ('pipe',)

    max_audio_length_ms = 240_000
    topk = 50
    temperature = 1.0
    cfg_scale = 1.5
    # Module erst bei Bedarf auf die GPU laden und danach wieder entladen (spart VRAM, langsamer)
    lazy_load = False

    def __init__(self, type='heartmula', model_name=None, device_mode='auto', lazy_load=None):
        super().__init__(type)
        self.hf_init(model_name, device_mode)
        if lazy_load is not None:
            self.lazy_load = bool(lazy_load)
        self.define_vars('mula_model', 'codec_model', 'max_audio_length_ms', 'topk', 'temperature', 'cfg_scale',
                         'lazy_load')
        self.define_sig(d.PStr('op', 'generate'), d.Param('lyrics'), d.PStr('tags', 'piano,happy'),
                        d.PStr('filename', 'music.mp3'))\
            .set_description("Erzeugt Musik aus Songtext und Tags und speichert sie als Audiodatei")\
            .r_success(filename='str')

    def hf_models(self) -> list:
        return [m for m in (self.model_name, self.mula_model, self.codec_model) if m]

    def set_mula_model(self, value):
        if value != self.mula_model:
            self.unload()
            self.mula_model = value

    def set_codec_model(self, value):
        if value != self.codec_model:
            self.unload()
            self.codec_model = value

    def set_lazy_load(self, value):
        value = value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'yes', 'on')
        if value != self.lazy_load:
            self.unload()
            self.lazy_load = value

    def load_model(self):
        import os
        import torch
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        from heartlib import HeartMuLaGenPipeline
        from heartlib.pipelines.music_generation import HeartMuLaGenConfig
        gen_path = snapshot_download(self.model_name)
        mula_path = snapshot_download(self.mula_model)
        codec_path = snapshot_download(self.codec_model)
        device = torch.device(self.device)
        self.pipe = HeartMuLaGenPipeline(
            heartmula_path=mula_path,
            heartcodec_path=codec_path,
            heartmula_device=device,
            heartcodec_device=device,
            # bf16 für HeartCodec verschlechtert laut heartlib die Audioqualität
            heartmula_dtype=torch.float32 if self.device == 'cpu' else torch.bfloat16,
            heartcodec_dtype=torch.float32,
            lazy_load=self.lazy_load and self.device != 'cpu',
            muq_mulan=None,
            text_tokenizer=Tokenizer.from_file(os.path.join(gen_path, 'tokenizer.json')),
            config=HeartMuLaGenConfig.from_file(os.path.join(gen_path, 'gen_config.json')),
        )

    def generate(self, lyrics, tags, output_file='music.mp3', max_audio_length_ms=None, topk=None,
                 temperature=None, cfg_scale=None):
        import torch
        self.load()
        with self.hf_lock, torch.no_grad():
            self.pipe(
                {'lyrics': lyrics, 'tags': tags},
                max_audio_length_ms=int(max_audio_length_ms or self.max_audio_length_ms),
                save_path=output_file,
                topk=int(topk or self.topk),
                temperature=float(temperature or self.temperature),
                cfg_scale=float(cfg_scale or self.cfg_scale),
            )
        return output_file

    def execute_generate(self, data):
        filename = self.generate(data['lyrics'], data.get('tags', ''), data.get('filename', 'music.mp3'),
                                 data.get('max_audio_length_ms'), data.get('topk'), data.get('temperature'),
                                 data.get('cfg_scale'))
        return self.success(filename=filename)
