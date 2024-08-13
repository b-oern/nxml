
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
import time
class Clip:

    ffmpeg = '/usr/bin/ffmpeg'

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
        cmd = self.ffmpeg + " -framerate 30 -pattern_type glob -i *.png -c:v libx264 -pix_fmt yuv420p out.mp4"

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
        if isinstance(image, str):
            image = Image.open(image)
        self.img = image
        self.x = x
        self.y = y

    def render_frame(self, image, t):
        image.paste(self.img, (self.x, self.y))
        # https://note.nkmk.me/en/python-pillow-paste/

class HorizontalAnimatedImage(ImageClip):
    def __init__(self, image, width=1920, height=1080, duration=10):
        super().__init__(image, 0, 0, width, height, duration)

    def render_frame(self, image, t):
        size = self.img.getbbox()
        w = self.width - size[3]
        percent = t/self.duration
        diff = int(percent * w)
        #print(f"{diff} 0 {diff+self.width}")
        img = self.img.crop((diff, 0, diff+self.width, self.height)) #  (left, upper, right, lower)-
        image.paste(img)

class AngularAnimatedImage(ImageClip):
    def __init__(self, image, width=1920, height=1080, duration=10):
        super().__init__(image, 0, 0, width, height, duration)

    def render_frame(self, image, t):
        size = self.img.getbbox()
        w = self.width - size[3]
        h = self.height - size[4]
        percent = t/self.duration
        diffX = int(percent * w)
        diffY = int(percent * h)
        img = self.img.crop((diffX, diffY, diffX+self.width, diffY+self.height)) #  (left, upper, right, lower)-
        image.paste(img)

class VideoClip(Clip):
    def __init__(self, file, fps=30):
        self.file = file
        self.fps = fps
        # TODO conver
        super().__init__()

    def cmd(self, workdir):
        # %04d
        c = 'ffmpeg -i '+self.file+' einzelbild%d.jpg'

class ConcatClip(Clip):
    def __init__(self, clips):
        super().__init__(width=clips[0].width, height=clips[0].height, duration=clips[0].duration)

