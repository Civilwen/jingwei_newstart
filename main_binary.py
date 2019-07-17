import argparse
import torch
import torch.optim as optim
import os
import pandas as pd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset_binary import AgricultureDataset
from dice_loss import BinaryDice
from unet import Unet_ResNet34 as Unet
from PIL import Image
import time


def save_checkpoint(state, path):
    torch.save(state, path)


def train(model, device, train_loader, optimizer, epoch, History: dict):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape)

        '''
         shape of test-data:torch.Size([1, 3, 512, 512]),test-target:torch.Size([1, 512, 512])
        shape:torch.Size([16, 4, 512, 512])  16是batch_size,4是三分类+一个背景       
        '''
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        '''
        weight表示4种分类各自在loss中占有的权重
        NLLLoss的结果就是把上面的输出output与Label对应的那个值(索引)拿出来，再去掉负号，再求均值。
        log_softmax+NLLLoss的操作可由CrossEntropyLoss直接合并        
        '''
        output = F.sigmoid(output)
        loss = BinaryDice(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:  # 每50个batchs打印一下训练状态
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    if epoch % 1 == 0:
        img_origin = (data.detach()[0, :, :, :].cpu().numpy() * 255).astype(np.uint8)
        img_origin = np.transpose(img_origin, (1, 2, 0))
        img_origin = Image.fromarray(img_origin)
        img_pre = output.detach()[0, :, :, :].cpu()
        img_pre = np.around(img_pre.byte().numpy()) * 220
        img_pre = Image.fromarray(img_pre)
        img_real = target.detach()[0, :, :, :].cpu()
        img_real = np.around(img_real.byte().numpy()) * 220
        img_real = Image.fromarray(img_real)
        img_origin.save('./results/tmp/img{}.png'.format(epoch))
        img_pre.save('./results/tmp/predict{}.png'.format(epoch))
        img_real.save('./results/tmp/real{}.png'.format(epoch))


def evaluate(model, device, data_loader, data_catag, History: dict, class_num=4):
    model.eval()
    eval_loss = 0
    max_batch = len(data_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            output = F.sigmoid(output)
            loss = BinaryDice(output, target).to('cuda')
            eval_loss += loss

    eval_loss = eval_loss / max_batch
    eval_acc = 1 - eval_loss

    History[data_catag + "_loss"].append(eval_loss)
    History[data_catag + "_acc"].append(eval_acc)
    print("{} Loss is {:.6f}, mean precision is: {:.4f}".format(data_catag, eval_loss, eval_acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--class_num', type=int, default=4, metavar='N',
                        help='input class number of label including background (default: 4)')
    parser.add_argument('--class_name', type=1, default=1, metavar='N',
                        help='input the class name which you want to seg')
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--results_path', type=str,
                        help='input your path to save the results')
    parser.add_argument('--image_path', type=str,
                        help='input your path to load the image')
    parser.add_argument('--label_path', type=str,
                        help='input your path to load the label')
    parser.add_argument('--data_csv_path', type=str,
                        help='input your path to load the date_cvs file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    if not os.path.exists(args.results_path + "_" + args.class_num):
        os.mkdir(args.results_path + "_" + args.class_name)
    if not os.path.exists(args.results_path + os.sep + "checkpoint_path"):
        os.mkdir(args.results_path + os.sep + "checkpoint_path")
    if not os.path.exists(args.results_path + os.sep + "model"):
        os.mkdir(args.results_path + os.sep + "model")
    if not os.path.exists(args.results_path + os.sep + 'loss_curve'):
        os.mkdir(args.results_path + os.sep + 'loss_curve')
    if not os.path.exists(args.results_path + os.sep + 'tmp'):
        os.mkdir(args.results_path + os.sep + 'tmp')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :', device)

    start_epoch = 0
    best_precision = 0

    History = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    model = Unet(args.class_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    resume = False
    # 4.4 restart the training process
    if resume:
        checkpoint = torch.load(args.results_path + os.sep + "checkpoint_path" + os.sep + "_checkpoint.pth.tar")
        start_epoch = checkpoint["epoch"]
        best_precision = checkpoint["best_precision"]
        model.load_state_dict(checkpoint["state_dict"])
        History = checkpoint["History"]
        print("---------------loading checkpoint----------------")
        optimizer.load_state_dict(checkpoint["optimizer"])

    data_list = pd.read_csv(args.data_csv_path)
    data_list = data_list.sample(frac=1)

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(AgricultureDataset(image_path=args.image_path,
                                                                  label_path=args.label_path,
                                                                  datalist=data_list,
                                                                  class_name=args.class_name,
                                                                  mode="train",
                                                                  train_ratio=0.8),
                                               batch_size=args.train_batch_size,
                                               shuffle=True,
                                               drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(AgricultureDataset(image_path=args.image_path,
                                                                  label_path=args.label_path,
                                                                  datalist=data_list,
                                                                  class_name=args.class_name,
                                                                  mode="valid",
                                                                  is_aug=False),
                                               batch_size=args.test_batch_size,
                                               shuffle=True,
                                               drop_last=True, **kwargs)

    for epoch in range(start_epoch, args.epochs + 1):

        train(model, device, train_loader, optimizer, epoch, History)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_precision": best_precision,
            "optimizer": optimizer.state_dict(),
            "History": History
        }, args.results_path + os.sep + "checkpoint_path" + os.sep + "/_checkpoint.pth.tar")

        if (epoch + 0) % 1 == 0:
            print("epoch:{}--------------------\n".format(epoch))

            evaluate(model, device, train_loader, "train", History, args.class_num)
            evaluate(model, device, valid_loader, "val", History, args.class_num)

            torch.save(model,
                       args.results_path + os.sep + "model" + os.sep + "model_epoch{}.pth".format(epoch))
            if History["val_loss"][-1] < best_precision:
                best_precision = History["val_loss"][-1]
                torch.save(model, args.results_path + os.sep + "model" + os.sep + "model_best.pth")
            plt.style.use("ggplot")
            plt.figure()
            N = epoch + 1
            plt.plot(np.arange(0, N), History["train_loss"], label="train_loss")
            plt.plot(np.arange(0, N), History["val_loss"], label="val_loss")
            plt.plot(np.arange(0, N), History["train_acc"], label="train_acc")
            plt.plot(np.arange(0, N), History["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy on Unet Satellite Seg")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig(args.results_path + os.sep + 'loss_curve' + os.sep + 'curve_{}.jpg'.format(epoch))
            plt.close()


if __name__ == '__main__':
    main()
