"""
Vision-Package.

- nxml.vision.face: FaceSimilarity
- nxml.vision.slam: SlamRunner (monokulares Visual SLAM, Kamerapose und Punktwolke aus Video/Stream)
- nxml.vision.pointcloud: PointCloudRunner (3D-Ansicht von Punktwolken, PLY/XYZ und SLAM-Karten)
- nxml.vision.ocr: BonReader (Kassenbon -> Positionen, Preise, Händler),
  DeepSeekOcr (DeepSeek-OCR über einen isolierten vLLM-Server)
"""

# Rückwärtskompatibilität: nxml.vision:FaceSimilarity, nxml.vision:ComfyUi
# Lazy, damit `import nxml.vision` nicht sofort torch/cv2 lädt.


def __getattr__(name):
    if name == 'FaceSimilarity':
        from nxml.vision.face import FaceSimilarity
        return FaceSimilarity
    if name == 'SlamRunner':
        from nxml.vision.slam import SlamRunner
        return SlamRunner
    if name == 'PointCloudRunner':
        from nxml.vision.pointcloud import PointCloudRunner
        return PointCloudRunner
    if name == 'BonReader':
        from nxml.vision.ocr import BonReader
        return BonReader
    if name == 'DeepSeekOcr':
        from nxml.vision.ocr import DeepSeekOcr
        return DeepSeekOcr
    if name in ('ComfyUi', 'Position'):
        from nxml import comfyui
        return getattr(comfyui, name)
    raise AttributeError(f"module 'nxml.vision' has no attribute {name!r}")


__all__ = ['FaceSimilarity', 'SlamRunner', 'PointCloudRunner', 'BonReader', 'DeepSeekOcr', 'ComfyUi']
