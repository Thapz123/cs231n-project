from PIL import Image
import os
import shutil


_size299 = (299,299)
SMALLEST_DIM = 400
OG_PATH = './originals_full/' #change depending dir containing orignals
PH_PATH = './photoshops_full/' #change depending on dir containing photoshops
PH_RESIZED_PATH = './photoshops_resized/'
OG_RESIZED_PATH = './originals_resized/'
SIZE_THRESHOLD = 1000 #1000bytes

'''Setup The Directories Containing Resized Images'''
if(os.path.exists(OG_RESIZED_PATH)):
    shutil.rmtree(OG_RESIZED_PATH)
if(os.path.exists(PH_RESIZED_PATH)):
    shutil.rmtree(PH_RESIZED_PATH)
    
os.mkdir(OG_RESIZED_PATH)
os.mkdir(PH_RESIZED_PATH)

def get_resize_dims(im):
    width, height = im.size
    maxwidth = 600
    maxheight = 600
    i = min(maxwidth/width, maxheight/height)
    a = max(maxwidth/width, maxheight/height)
    dims = (width*a/i, height*a/i)
    return dims

'''Resize Originals'''
for f in os.listdir(OG_PATH):
    if(os.stat(OG_PATH+f).st_size > SIZE_THRESHOLD):
        try:
            #try to open the image
            i = Image.open(OG_PATH + f)      
            dims = get_resize_dims(i)
            i.thumbnail(dims, Image.ANTIALIAS)
            i.save(OG_RESIZED_PATH+f)
        except IOError:
            #if image is broken, remove it.
            os.remove(OG_PATH + f) 
    else:
        os.remove(OG_PATH + f) 


'''Resize Photoshops'''
for f in os.listdir(PH_PATH):
    if(os.stat(PH_PATH+f).st_size > SIZE_THRESHOLD):
        try:
            i = Image.open(PH_PATH+f)
            dims = get_resize_dims(i)
            i.thumbnail(dims, Image.ANTIALIAS)
            i.save(PH_RESIZED_PATH+f)
        except IOError:
            os.remove(PH_PATH + f)
    else:
        os.remove(PH_PATH + f) 


