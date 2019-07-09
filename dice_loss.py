import torch.nn as nn


class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss.self).__init__()

    def forward(self, input, target):
        N = target.size(0)  # N = batchsize
        smooth = 1

        # 将input和output resize为batch_size行的二维矩阵
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2. * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):
        C = input.shape[1]

        dice = Dice_Loss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


MulDiceLoss = MulticlassDiceLoss()
