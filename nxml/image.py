#
# Project: nxml
#

from nwebclient import runner as r
from nwebclient import NWebClient


class ImageSimilarity(r.ImageExecutor):
    """
      DocMap in nwebclient.nc
    """

    MODULES = ['opencv-python', 'scikit-image']
    type = 'image_similarity'

    def compareImageSSIM(self, image_a, image_b):
        import cv2
        from skimage import metrics
        import numpy
        # Load images
        #image1 = cv2.imread(image_a)
        #image2 = cv2.imread(image_b)
        image1 = cv2.cvtColor(numpy.array(image_a), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(numpy.array(image_b), cv2.COLOR_RGB2BGR)
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
        print(image1.shape, image2.shape)
        # Convert images to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Calculate SSIM
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
        #print(f"SSIM Score: ", round(ssim_score[0], 2))
        return round(ssim_score[0], 2)

    def searchSimilar(self, image, data):
        n = NWebClient(None)
        q = 'kind=image&limit=50'
        docs = n.docs(q)
        self.info(f"Calculating Similarity with {len(docs)} Images")
        for d in docs:
            if d.is_image():
                img_b = d.as_image()
                d.similarity = self.compareImageSSIM(image, img_b)
                self.info("Similarity: "+str(d.similarity))
        docs.sort(key=lambda x: x.similarity, reverse=True)
        i = 1
        for d in docs:
            self.info(f"{i}: {d.name()} {d.id()}")
            i += 1
            if i > 5:
                break

    def executeImage(self, image, data):
        b = self.get_image('b', data)
        if b is not None:
            score = self.compareImageSSIM(image, b)
            return {'success': True, 'score': score}
        if 'search' in data:
            return self.searchSimilar(image, data)
        else:
            return {}

