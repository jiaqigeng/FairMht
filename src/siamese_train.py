import torch
from siamese_model import SiameseNet, EmbeddingNet
from siamese_loss import ContrastiveLoss
from siamese_dataloader import SiameseDataset
from torch.utils.data import Dataset, DataLoader
import random
import wandb
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm


def train(net, optimizer, criterion1, criterion2, num_epoch, train_dataloader):
    net.train()
    running_loss = 0
    running_acc = 0
    for epoch in range(num_epoch):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            optimizer.zero_grad()
            output1, output2, out = net(img0, img1)

            loss = criterion2(out, label.squeeze().long())

            loss.backward()
            optimizer.step()

            label = torch.reshape(label, (-1,))
            _, predicted = torch.max(out, 1)
            acc = torch.eq(predicted, label).sum() / label.size(0)
            running_acc += acc
            running_loss += loss.item()

            wandb.log({'loss': loss.item(), 'acc': acc})
            running_loss, running_acc = 0, 0


def test(net, test_dataloader):
    net.eval()
    
    running_corrects = 0
    running_counts = 0
    
    for i, data in enumerate(tqdm(test_dataloader)):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

        output1, output2, out = net(img0, img1)
        label = torch.reshape(label, (-1,))

        _, predicted = torch.max(out, 1)
        running_corrects += torch.eq(predicted, label).sum()
        running_counts += label.size(0)

    print('Accuracy: {}'.format(running_corrects / running_counts))


def main():
    wandb.init(project="16824-project")
    # Declare Siamese Network
    embedding_net = EmbeddingNet()
    net = SiameseNet(embedding_net).cuda()
    # Decalre Loss Function
    criterion1 = ContrastiveLoss(margin=1.)
    criterion2 = nn.CrossEntropyLoss()

    # Declare Optimizer
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    data_list = []
    with open("people.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            anchor, neg, pos = line.split(" ")
            data_list.append([anchor, neg, 0.])
            data_list.append([anchor, pos, 1.])

    random.shuffle(data_list)

    siamese_dataset = SiameseDataset(data_list)
    siamese_dataset_test = SiameseDataset(data_list, mode='test')
    train_dataloader = DataLoader(siamese_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(
        siamese_dataset_test, batch_size=16, shuffle=False, num_workers=8)

    if 0:
        train(net, optimizer, criterion1, criterion2, num_epoch=20, train_dataloader=train_dataloader)
        torch.save(net.state_dict(), "siamese_model.pth")
        print("Model Saved Successfully")

    if 1:
        net.load_state_dict(torch.load("siamese_model.pth"))
        test(net, test_dataloader)


if __name__ == '__main__':
    main()
