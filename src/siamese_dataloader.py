import os
import random
from PIL import Image
from torchvision import transforms
import torch


class SiameseDataset:
    def __init__(self, data_list, mode='train'):
        # used to prepare the labels and images path
        self.mode = mode
        self.data_list = data_list
        
        self.img_dic = {}
        self.people_ids = [i for i in range(22)]
        for people_id in self.people_ids:
            self.img_dic[people_id] = os.listdir(f'People/{people_id}')
        
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        # getting the image path
        if self.mode == 'train':
            id1 = random.choice(self.people_ids)
            id2 = id1
            if random.random() > 0.5:
                while id1 == id2:
                    id2 = random.choice(self.people_ids)
            
            image1_path = f'People/{id1}/' + random.choice(self.img_dic[id1])
            image2_path = f'People/{id2}/' + random.choice(self.img_dic[id2])
            label = (id1 == id2) * 1.
        else:
            image1_path = self.data_list[index][0]
            image2_path = self.data_list[index][1]
            label = self.data_list[index][2]

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        return img0, img1, torch.FloatTensor([int(label)])

    def __len__(self):
        # return len(self.data_list)
        data_len = 10000 if self.mode == 'train' else len(self.data_list)
        return data_len

