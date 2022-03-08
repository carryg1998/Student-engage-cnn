import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from model.CNN import CNNnet
from utils.utils import get_val_data
from utils.dataloader import StuEnDataset, StuEnDataset_collect_fn

def fit_one_train(epoch, train_loader, model, criterion, optimizer, sum_train, sum_val, val_loader, val_data, val_label, cuda):
    train_losses = []
    print("Start training")
    for e in range(epoch):
        with tqdm(total=sum_train, desc=f'Epoch {e + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
            train_loss = 0
            model.train()
            for X, y in train_loader:
                X = torch.from_numpy(X).type(torch.FloatTensor).permute(0, 3, 1, 2)
                y = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in y]
                y = torch.tensor(y, dtype=torch.long)
                if cuda:
                    X = X.cuda()
                    y = y.cuda()
                X_data = Variable(X)
                out = model(X_data)

                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)

            print("Epoch", e+1, " loss:", train_loss / len(train_data))

        sum_acc = 0
        c = 0

        with tqdm(total=sum_val, desc=f'Epoch {e + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
            val_loss = 0
            model.eval()
            for X, y in val_loader:
                X = torch.from_numpy(X).type(torch.FloatTensor).permute(0, 3, 1, 2)
                y = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in y]
                y = torch.tensor(y, dtype=torch.long)
                if cuda:
                    X = X.cuda()
                    y = y.cuda()
                X_data = Variable(X)
                out = model(X_data)

                loss = criterion(out, y)

                acc = np.mean((torch.argmax(out, 1).cpu().numpy() == y.cpu().numpy()))
                sum_acc += acc
                c = c + 1

                val_loss += loss.item()
                pbar.update(1)

        torch.save(model.state_dict(), "logs/Epoch" + str(e + 1) + ".pth")
        print("Epoch", e+1, " acc:", sum_acc/c)

        train_losses.append(train_loss / len(train_data))
    return acc, model
    # return acc, f1_score, f1_score_mean, model

if __name__ == "__main__":
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#
    #GPU训练
    cuda = False
    device = torch.device("cuda:0")

    learning_rate = 0.001
    epoch = 5
    batch_size = 4
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#

    model = CNNnet(6)

    print("Cuda:", cuda)
    print("Learning rate:", learning_rate)
    print("Epoch:", epoch)
    print("Batch size:", batch_size)

    train_txt = "train_lines.txt"
    val_txt = "val_lines.txt"
    with open(train_txt) as f:
        train_lines = f.readlines()
    with open(val_txt) as f:
        val_lines = f.readlines()

    sum_train = len(train_lines) // batch_size
    sum_val = len(val_lines) // batch_size

    print("Loading data...")
    train_data = StuEnDataset(train_lines, [256, 256])
    train_gen = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                           drop_last=True, collate_fn=StuEnDataset_collect_fn)

    val_data = StuEnDataset(val_lines, [256, 256])
    val_gen = DataLoader(val_data, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                           drop_last=True, collate_fn=StuEnDataset_collect_fn)

    val_data, val_label = get_val_data(val_lines, [256, 256])
    val_data = Variable(torch.tensor(val_data))
    print("Finished!")

    if cuda:
        model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    acc, model = fit_one_train(epoch, train_gen, model, criterion, optimizer, sum_train, sum_val, val_gen, val_data,
                               val_label, cuda)
    print("精度为：", acc)
    # print("每个类的F1值如下：", f1_score)
    # print("平均F1值为：", f1_score_mean)+