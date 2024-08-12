
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
import time
class Clip:
    def __init__(self, width=1920, height=1080, duration=10):
        self.width = width
        self.height = height
        self.duration = duration

    def render_frame(self, image, t):
        pass

    def render(self, prefix, frame_offset, fps=30):
        frame_count = self.duration * fps
        for i in range(0, frame_count):
            image = Image.new(mode="RGB", size=(self.width, self.height))
            self.render_frame(image, i/fps)
            index = frame_offset + i
            name = f"{index:04d}"
            image.save(prefix + name + '.png')
        print("Done")

    def save_video(self, name='video'):
        self.render('vid', 0, 30)
        d = os.getcwd()
        cmd = "ffmpeg -framerate 30 -pattern_type glob -i '"+d+"/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4"

        print("D: " + d)
        time.sleep(5)
        r = subprocess.run(cmd.split(' '), cwd=d, stdout=subprocess.PIPE)
        print(r.stdout)

class CompositeClip(Clip):
    def __init__(self, clips=[]):
        super().__init__(width=clips[0].width, height=clips[0].height, duration=clips[0].duration)
        self.clips = clips

    def render_frame(self, image, t, fps=30):
        for clip in self.clips:
            clip.render_frame(image, t, fps)

class TextClip(Clip):
    def __init__(self, text="Text", x=10, y=10, width=1920, height=1080, duration=10):
        super().__init__(width, height, duration)
        self.text = text
        self.x = x
        self.y = y
        self.font = ImageFont.load_default(72)

    def render_frame(self, image, t):
        draw = ImageDraw.Draw(image)
        draw.text((self.x, self.y), self.text, font=self.font)

class ImageClip(Clip):
    def __init__(self, image, x=10, y=10, width=1920, height=1080, duration=10):
        super().__init__(width, height, duration)
        self.img = image
        self.x = x
        self.y = y

    def render_frame(self, image, t):
        image.paste(self.img, (self.x,self.y))
        # https://note.nkmk.me/en/python-pillow-paste/

class AnimatedImage(Clip):
    def __init__(self, image, width=1920, height=1080, duration=10):
        super().__init__(width, height, duration)
        if isinstance(image, str):
            image = Image.open(image)
        self.img = image

    def render_frame(self, image, t):
        size = self.img.getbbox()
        w = size[3] - self.width
        percent = t/self.duration
        diff = int(percent * w)
        img = self.img.crop((diff, 0, self.width, self.height)) #  (left, upper, right, lower)-
        image.paste(img)

class ConcatClip(Clip):
    def __init__(self, clips):
        super().__init__(width=clips[0].width, height=clips[0].height, duration=clips[0].duration)

