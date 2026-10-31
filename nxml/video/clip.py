#
# Project: nxml
#
# Videos aus Clips rendern (Pillow + ffmpeg) und Einzelbilder aus Videos lesen
#
# clip = CompositeClip([ImageClip('bg.png', 0, 0), TextClip("Hallo", 100, 100)])
# clip.save_video('hallo.mp4')
#

import io
import json
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

__all__ = ['VideoFrames', 'Clip', 'CompositeClip', 'TextClip', 'ImageClip', 'HorizontalAnimatedImage',
           'AngularAnimatedImage', 'VideoClip', 'ConcatClip']


class VideoFrames:
    """
    Liest Einzelbilder aus einem Video, mit OpenCV (schnell) oder, falls nicht installiert, mit ffmpeg/ffprobe.
    """

    def __init__(self, path, ffmpeg='ffmpeg'):
        self.path = path
        self.ffmpeg = ffmpeg
        self.cap = None
        try:
            import cv2
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self.cap = None
        except ImportError:
            pass

    def ffprobe(self):
        exe = self.ffmpeg[:-6] + 'ffprobe' if self.ffmpeg.endswith('ffmpeg') else 'ffprobe'
        out = subprocess.run([exe, '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=r_frame_rate:format=duration', '-of', 'json', self.path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        info = json.loads(out)
        num, _, den = info['streams'][0].get('r_frame_rate', '0/1').partition('/')
        return float(info['format']['duration']), float(num) / float(den or 1)

    def info(self) -> dict:
        if self.cap is not None:
            import cv2
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
            frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration = frames / fps if fps > 0 else 0
        else:
            duration, fps = self.ffprobe()
        # Größe vom ersten Bild, berücksichtigt die Rotation aus den Metadaten
        image = self.frame(0)
        return {'duration': round(duration, 3), 'fps': round(fps, 3), 'width': image.width, 'height': image.height}

    def frame(self, t):
        """ :return: PIL-Bild zum Zeitpunkt t (Sekunden) """
        t = max(0.0, float(t))
        if self.cap is not None:
            import cv2
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = self.cap.read()
            if not ok and t > 0:
                # hinter dem letzten Bild: letztes lesbares Bild nehmen
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
                ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f'No frame at {t}s')
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = subprocess.run([self.ffmpeg, '-v', 'error', '-ss', str(t), '-i', self.path, '-frames:v', '1',
                              '-f', 'image2pipe', '-vcodec', 'png', '-'], stdout=subprocess.PIPE, check=True).stdout
        if len(out) == 0:
            raise ValueError(f'No frame at {t}s')
        return Image.open(io.BytesIO(out)).convert('RGB')

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def paste(target, image, pos=(0, 0)):
    """ Einfügen mit Transparenz, falls das Bild einen Alphakanal hat """
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGBA')
        target.paste(image, pos, image)
    else:
        target.paste(image, pos)


class Clip:
    """
    Basis: render_frame(image, t) zeichnet den Clip zum Zeitpunkt t (Sekunden) in image.
    """

    ffmpeg = 'ffmpeg'

    def __init__(self, width=1920, height=1080, duration=10):
        self.width = width
        self.height = height
        self.duration = duration

    def render_frame(self, image, t):
        pass

    def frame_count(self, fps=30):
        return int(round(self.duration * fps))

    def render(self, prefix, frame_offset=0, fps=30):
        """ Speichert die Einzelbilder als prefix0000.png, prefix0001.png, ... :return: Anzahl Bilder """
        count = self.frame_count(fps)
        for i in range(count):
            image = Image.new(mode="RGB", size=(self.width, self.height))
            self.render_frame(image, i / fps)
            image.save(f"{prefix}{frame_offset + i:04d}.png")
        return count

    def save_video(self, name='video', fps=30):
        """
            Rendert in einen temporären Ordner und erzeugt daraus das Video.
            :param name: Ausgabedatei, ohne Endung wird .mp4 angehängt
            :return: Pfad des Videos
        """
        if os.path.splitext(name)[1] == '':
            name += '.mp4'
        out = os.path.abspath(name)
        with tempfile.TemporaryDirectory() as folder:
            self.render(os.path.join(folder, 'frame'), 0, fps)
            cmd = [self.ffmpeg, '-y', '-v', 'error', '-framerate', str(fps), '-i', os.path.join(folder, 'frame%04d.png'),
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0:
                raise RuntimeError('ffmpeg failed: ' + r.stderr.decode(errors='replace'))
        return out


class CompositeClip(Clip):
    """ Clips übereinander, in der Reihenfolge der Liste (der erste liegt unten) """

    def __init__(self, clips):
        if len(clips) == 0:
            raise ValueError('CompositeClip needs at least one clip')
        super().__init__(width=clips[0].width, height=clips[0].height, duration=max(c.duration for c in clips))
        self.clips = clips

    def render_frame(self, image, t):
        for clip in self.clips:
            if t < clip.duration:
                clip.render_frame(image, t)


class TextClip(Clip):
    def __init__(self, text="Text", x=10, y=10, width=1920, height=1080, duration=10, size=72, color='white'):
        super().__init__(width, height, duration)
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.font = ImageFont.load_default(size)

    def render_frame(self, image, t):
        draw = ImageDraw.Draw(image)
        draw.text((self.x, self.y), self.text, font=self.font, fill=self.color)


class ImageClip(Clip):
    def __init__(self, image, x=10, y=10, width=1920, height=1080, duration=10):
        super().__init__(width, height, duration)
        if isinstance(image, str):
            image = Image.open(image)
        self.img = image
        self.x = x
        self.y = y

    def progress(self, t):
        """ 0..1 über die Dauer des Clips """
        return min(1.0, max(0.0, t / self.duration)) if self.duration > 0 else 0.0

    def render_frame(self, image, t):
        paste(image, self.img, (self.x, self.y))
        # https://note.nkmk.me/en/python-pillow-paste/


class HorizontalAnimatedImage(ImageClip):
    """ Schwenk von links nach rechts über ein Bild, das breiter als der Clip ist """

    def __init__(self, image, width=1920, height=1080, duration=10):
        super().__init__(image, 0, 0, width, height, duration)

    def render_frame(self, image, t):
        diff = int(self.progress(t) * max(0, self.img.width - self.width))
        paste(image, self.img.crop((diff, 0, diff + self.width, self.height)))  # (left, upper, right, lower)


class AngularAnimatedImage(ImageClip):
    """ Schwenk diagonal von links oben nach rechts unten """

    def __init__(self, image, width=1920, height=1080, duration=10):
        super().__init__(image, 0, 0, width, height, duration)

    def render_frame(self, image, t):
        p = self.progress(t)
        dx = int(p * max(0, self.img.width - self.width))
        dy = int(p * max(0, self.img.height - self.height))
        paste(image, self.img.crop((dx, dy, dx + self.width, dy + self.height)))


class VideoClip(Clip):
    """
    Ausschnitt eines vorhandenen Videos ab start, auf width x height skaliert (Standard: Größe des Videos).
    """

    def __init__(self, file, x=0, y=0, width=None, height=None, start=0, duration=None, ffmpeg='ffmpeg'):
        self.file = file
        self.x = x
        self.y = y
        self.start = start
        self.frames = VideoFrames(file, ffmpeg)
        info = self.frames.info()
        self.fps = info['fps']
        if duration is None:
            duration = max(0.0, info['duration'] - start)
        super().__init__(width or info['width'], height or info['height'], duration)
        self.ffmpeg = ffmpeg

    def render_frame(self, image, t):
        frame = self.frames.frame(self.start + t)
        if frame.size != (self.width, self.height):
            frame = frame.resize((self.width, self.height), Image.LANCZOS)
        paste(image, frame, (self.x, self.y))

    def close(self):
        self.frames.close()


class ConcatClip(Clip):
    """ Clips nacheinander, Größe vom ersten Clip """

    def __init__(self, clips):
        if len(clips) == 0:
            raise ValueError('ConcatClip needs at least one clip')
        super().__init__(width=clips[0].width, height=clips[0].height, duration=sum(c.duration for c in clips))
        self.clips = clips

    def render_frame(self, image, t):
        for clip in self.clips:
            if t < clip.duration or clip is self.clips[-1]:
                clip.render_frame(image, t)
                return
            t -= clip.duration
