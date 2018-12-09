from PIL import Image
from resizeimage import resizeimage


def resize(image):
	with open(image, 'r+b') as f:
	    with Image.open(f) as pic:
	        cover = resizeimage.resize_cover(pic, [227, 227])
	        cover.save(image, pic.format)

