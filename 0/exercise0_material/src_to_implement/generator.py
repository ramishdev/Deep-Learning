import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_batch_index = 0
        self.epoch = 0
        with open(label_path) as f:
            self.labels = json.load(f)
        self.image_paths = [os.path.join(file_path, img) for img in os.listdir(file_path) if img.endswith('.npy')]
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Class dictionary
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        #return images, labels
        if self.current_batch_index + self.batch_size > len(self.image_paths):
            remaining = len(self.image_paths) - self.current_batch_index
            indices = self.indices[self.current_batch_index:] 
            self.current_batch_index = 0
            self.epoch += 1
            if self.shuffle:
                np.random.shuffle(self.indices)
            if remaining < self.batch_size:
                indices = np.concatenate([indices, self.indices[:self.batch_size - remaining]])

        else:
            indices = self.indices[self.current_batch_index:self.current_batch_index + self.batch_size]
            self.current_batch_index += self.batch_size
        
        images = [self.augment(np.load(self.image_paths[i])) for i in indices]
        labels = [self.labels[os.path.basename(self.image_paths[i]).split('.')[0]] for i in indices]
        return np.array(images), np.array(labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring and np.random.rand() > 0.5:
            img = np.fliplr(img)
        if self.rotation:
            angle = np.random.choice([0, 90, 180, 270])
            img = transform.rotate(img, angle)
        img = transform.resize(img, self.image_size, anti_aliasing=True)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(zip(images, labels)):
            plt.subplot(self.batch_size//2, 3, i + 1)
            plt.imshow(image)
            plt.title(self.class_name(label))
            plt.axis('off')
        plt.show()