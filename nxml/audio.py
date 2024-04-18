

from nwebclient import runner as r


class LibRosa(r.BaseJobExecutor):
    MODULES = ['librosa']
    type='librosa'

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
