import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE = 256

data_transforms ={
    'train' : transforms.Compose([
        
        
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # resize the image to 299x299
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # loader for evaluation, no horizontal flip
    'val' : transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # resize the image to 299x299
        transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

}


class Pix2PixDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self,  data_dir_original, data_dir_photoshopped, direction,transform = data_transforms['train']):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir_photoshopped: (string) directory containing the photoshopped images
            data_dir_real: (string) directory containing the real images
            transform: (torchvision.transforms) transformation to apply on image
        """
        '''
        Notes:
        Would need to just save all the images and label them appropriately.
        Need to figure out how to just extract 1 photoshop transformation fo reach image for each 
        
        '''
        self.direction = direction
        cwd  = os.path.dirname(os.path.realpath(__file__))
        # Process Photoshopped Images
        data_dir_photoshopped = os.path.join(cwd, '..', data_dir_photoshopped)
        #print(data_dir_photoshopped)
        p_filenames = os.listdir(data_dir_photoshopped)
        temp_photo= [f for f in p_filenames if f.endswith('.jpg')]
        p_filenames = [os.path.join(data_dir_photoshopped, f) for f in temp_photo]
#         p_labels = [1 for i in range(len(p_filenames))]
        
        # Process Original Images
        data_dir_original = os.path.join(cwd,'..', data_dir_original)
        o_filenames = os.listdir(data_dir_original)
        temp_orig = [f for f in o_filenames if f.endswith('.jpg')]
        o_filenames = [os.path.join(data_dir_original, f) for f in temp_orig]
#         o_labels = [0 for i in range(len(o_filenames))]
        
        # Add both types together
        self.filenames = []
        for i, orig in enumerate(temp_orig):
            name = orig.split('.')[0]
            for j, photo in enumerate(temp_photo):
                if name in photo:
                    self.filenames.append((o_filenames[i], p_filenames[j]))
                    break
#         self.filenames = [(orig, photo)]
#         self.filenames = p_filenames + o_filenames
#         self.labels = p_labels + o_labels
        self.labels = [(0,1) for _ in range(len(self.filenames))]
        self.filenames, self.labels =  shuffle(self.filenames, self.labels, random_state=42)
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def check_image(self, file):
        image = Image.open(file)
        width, height = image.size
        if width<IMAGE_SIZE or height<IMAGE_SIZE:
            image = image.resize((IMAGE_SIZE+50, IMAGE_SIZE+50))
        if image.mode!="RGB":
            image = image.convert('RGB')
        return image
    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        o_file, p_file = self.filenames[idx]
        o_image = self.check_image(o_file)  # PIL image
        p_image = self.check_image(p_file)
        o_tensor = self.transform(o_image)
        p_tensor = self.transform(p_image)
        if self.direction == 'AtoB':
            sample = {'A': o_tensor, 'B': p_tensor}
        else:
            sample = {'B': o_tensor, 'A': p_tensor}
        return sample
