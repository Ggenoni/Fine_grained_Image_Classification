import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import numpy as np
import json


def get_data_flowers(batch_size_train, batch_size_test, num_workers, transform=None):

    if not transform:
        # Define the data transformations for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define the data transformations for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transform
        val_transform = transform

    # Download the Flowers102 dataset
    train_dataset = datasets.Flowers102(root='data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.Flowers102(root='data', split='val', download=True, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader



def calculate_mean_std(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count

    return mean, std

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self.image_paths = []
        self.labels = []

        # Create a list of image paths and their corresponding labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
        
        # Optionally save the class mapping to a file
        class_mapping = self.idx_to_class
        with open('class_mapping.json', 'w') as f:
            json.dump(class_mapping, f)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomTestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Create a list of image paths
        for img_name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_name)
            self.image_paths.append((img_path, os.path.splitext(img_name)[0]))  # Store both image path and ID
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, img_id = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, img_id

def create_dataloader(root_dir, batch_size=32, img_size=224, val_split=0.5, mode='train', transform=None):
    if mode == 'train':
        if transform is None:

            # Define the training and validation transformations
            train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

            val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        else:
            train_transform = transform
            val_transform = transform

        # Create an instance of the custom dataset with the training transformations
        full_dataset = CustomImageDataset(root_dir=root_dir, transform=train_transform)

        # Split the dataset into training and validation sets
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply the validation transformation to the validation dataset
        val_dataset.dataset.transform = val_transform

        # Create DataLoaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader

    elif mode == 'test':
        # Define the test transformations
        if transform is None:    

            test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                                std=[0.229, 0.224, 0.225])])  # Normalize with ImageNet std
        else:
            test_transform = transform

        # Create an instance of the custom test dataset with the test transformations
        test_dataset = CustomTestImageDataset(root_dir=root_dir, transform=test_transform)

        # Create a DataLoader for the test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return test_loader
