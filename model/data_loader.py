import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.utils import shuffle

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.RandomCrop(600),  # resize the image to 224x224 
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()  # transform it into a torch tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
])
# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.RandomCrop(600),  # resize the image to 224 
    transforms.ToTensor()  # transform it into a torch tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PhotoshopDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self,  data_dir_original, data_dir_photoshopped, transform = train_transformer):
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
        cwd  = os.path.dirname(os.path.realpath(__file__))
        # Process Photoshopped Images
        data_dir_photoshopped = os.path.join(cwd, '..', data_dir_photoshopped)
        #print(data_dir_photoshopped)
        p_filenames = os.listdir(data_dir_photoshopped)
        p_filenames = [os.path.join(data_dir_photoshopped, f) for f in p_filenames if f.endswith('.jpg')]
        p_labels = [1 for i in range(len(p_filenames))]
        
        # Process Original Images
        data_dir_original = os.path.join(cwd,'..', data_dir_original)
        o_filenames = os.listdir(data_dir_original)
        o_filenames = [os.path.join(data_dir_original, f) for f in o_filenames if f.endswith('.jpg')]
        o_labels = [0 for i in range(len(o_filenames))]
        
        # Add both types together
        self.filenames = p_filenames + o_filenames
        self.labels = p_labels + o_labels
        
        #randomly shuffle data
#         mapIndexPosition = list(zip(filenames, labels))
#         random.shuffle(mapIndexPosition)
#         self.filenames, self.labels = zip(*mapIndexPosition)
        self.filenames, self.labels =  shuffle(self.filenames, self.labels, random_state=42)
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        tensor = self.transform(image)
        #image = tensor
        sample = {'image': image,  'label':self.labels[idx], 'tensor':tensor}
        return sample


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
