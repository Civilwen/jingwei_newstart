import argparse
import torch
import torch.optim as optim
import os
import pandas as pd
from dataset import AgricultureDataset
from dice_loss import MulDiceLoss
from unet import Unet_ResNet34 as Unet
import time


def save_checkpoint(state, path):
    torch.save(state, path)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

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

        loss = MulDiceLoss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:  # 每50个batchs打印一下训练状态
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def evaluate(model, device, data_loader, data_categ: str, History: dict, class_num=4):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    max_batch = len(data_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = MulDiceLoss().to('cuda')(output, target)
            eval_loss += loss

            # predict = torch.argmax(output, 1).byte()  # argmax中dim表示要消失的那个维度，即返回该维度上值最大的索引值  完美解决像素表示问题
            # target = target.byte()

    eval_acc = eval_acc / max_batch
    eval_loss = eval_loss / max_batch

    History[data_categ + "_loss"].append(eval_loss)
    History[data_categ + "_acc"].append(eval_acc)
    print("{} Loss is {:.6f}, mean precision is: {:.4f}%".format(data_categ, eval_loss, eval_acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--class_num', type=int, default=4, metavar='N',
                        help='input class number of label including background (default: 4)')
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
    parser.add_argument('--data_cvs_path', type=str,
                        help='input your path to load the date_cvs file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    if os.path.exists(args.results_path):
        os.mkdir(args.results_path)
    if os.path.exists(args.results_path + os.sep + "checkpoint_path"):
        os.mkdir(args.results_path + os.sep + "checkpoint_path")
    if os.path.exists(args.results_path + os.sep + "model"):
        os.mkdir(args.results_path + os.sep + "model")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :', device)

    start_epoch = 0
    best_precision = 0

    History = {"train_loss": [],
               # "train_acc": [],
               "val_loss": [],
               # "val_acc": []
               }

    model = Unet(args.class_num).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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
                                                                  mode="train",
                                                                  train_ratio=0.7),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(AgricultureDataset(image_path=args.image_path,
                                                                  label_path=args.label_path,
                                                                  datalist=data_list,
                                                                  mode="valid"),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True, **kwargs)

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_precision": best_precision,
            "optimizer": optimizer.state_dict(),
            "History": History
        }, args.results_path + os.sep + "checkpoint_path" + os.sep + "/_checkpoint.pth.tar")

        if epoch % 3 == 0:
            print("epoch:{}--------------------\n".format(epoch))
            evaluate(model=model, device=device, data_loader=train_loader,
                     data_categ="train", History=History, class_num=args.class_num)
            evaluate(model=model, device=device, data_loader=valid_loader,
                     data_categ="val", History=History, class_num=args.class_num)
            torch.save(model,
                       args.results_path + os.sep + "model" + os.sep + "model_epoch{}.pth".format(epoch))
            if History["val_loss"][-1] < best_precision:
                best_precision = History["val_loss"][-1]
                torch.save(model, args.results_path + os.sep + "model" + os.sep + "model_best.pth")


if __name__ == '__main__':
    main()
