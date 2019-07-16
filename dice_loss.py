import torch.nn as nn


class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, _output, _target):
        N = _target.size(0)  # N = batchsize
        smooth = 0.00001

        # 将input和output resize为batch_size行的二维矩阵
        output_flat = _output.view(N, -1)
        target_flat = _target.view(N, -1)

        intersection = output_flat * target_flat

        loss = (2. * intersection.sum(1) + smooth) / (output_flat.sum(1) + target_flat.sum(1) + smooth)

        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, _output, _target, weights=None):
        C = _output.shape[1]
        #print("diceLoss _output_shape:", _output.shape)
        #print("diceLoss _target_shape:", _target.shape)

        dice = Dice_Loss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(_output[:, i], _target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


MulDiceLoss = MulticlassDiceLoss()
