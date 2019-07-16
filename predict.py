import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000

def get_predict(img, model, device):
    img = np.array(img) / 255
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    output = model(img)
    output = F.softmax(output, dim=1)
    predict = torch.argmax(output, 1).byte().cpu().numpy()
    print(predict.shape)
    return predict


def main(model_path, image_path, predict_path, epoch_num):
    use_cuda = True
    model = torch.load(model_path)  # torch.save(model,'./tmp/model{}'.format(epoch))
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    files = os.listdir(image_path)
    for file in files:
        file_name = file.split(".")

        image = Image.open(image_path + os.sep + file)
        _width, _height = image.size
        _predict = np.zeros([_height, _width], np.uint8)
        crop_size = 1024
        stride = 512
        for _y in range(0, _height - crop_size, stride):
            for _x in range(0, _width - crop_size, stride):
                img = np.array(image.crop((_x, _y, _x + crop_size, _y + crop_size)))
                #print(img.shape)
                img = img[:, :, 0:3]
                img = np.transpose(img, (2,0,1))
                if (img.sum() == 0): continue
                _predict[_y:_y + crop_size, _x:_x + crop_size] = get_predict(img, model=model, device=device)

            img = np.array(image.crop((_width - crop_size, _y, _width, _y + crop_size)))
            img = img[:, :, 0:3]
            img = np.transpose(img, (2,0,1))
            if (img.sum() != 0):
                _predict[ _y:_y + crop_size, (_width - crop_size):_width] = get_predict(img, model=model, device=device)

        for _x in range(0, _width - crop_size, stride):
            img = np.array(image.crop((_x, _height - crop_size, _x + crop_size, _height)))            
            img = img[:, :, 0:3]
            img = np.transpose(img, (2,0,1))

            if (img.sum() != 0):
                _predict[(_height - crop_size):_height, _x:_x + crop_size] = get_predict(img, model=model, device=device)

        img = np.array(image.crop((_width - crop_size, _height - crop_size, _width, _height)))
        img = img[:, :, 0:3]
        img = np.transpose(img, (2,0,1))
        if (img.sum() != 0):
            _predict[(_height - crop_size):_height, (_width - crop_size):_width] = get_predict(img, model=model, device=device)

        im = Image.fromarray(_predict)
        im.save(predict_path + os.sep + file_name[0] + '_' + epoch_num + "_predict.png")
        im_vis = Image.fromarray(_predict * 80)
        im_vis.save(predict_path + os.sep + file_name[0] + '_' + epoch_num + "_predict_vis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jingwei predict script")
    parser.add_argument('--model_path', type=str, help='Choose the model you want to use for prediction')
    parser.add_argument('--image_path', type=str, help='Image path you need to predict')
    parser.add_argument('--predict_path', type=str, help='Path where you want to save your results')
    parser.add_argument('--epoch_num', type=str, help='Path where you want to save your results')

    args = parser.parse_args()

    if not os.path.exists(args.predict_path):
        os.mkdir(args.predict_path)

    main(args.model_path, args.image_path, args.predict_path, args.epoch_num)
