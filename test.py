import argparse
from models.MVCNN import MVCNN, SVCNN
import torch
import os
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
class Test():
    def __init__(self, model_path='/home/spring/mnt/sda3/code/mvcnn_pytorch-master/mvcnn_stage_2/mvcnn/model-00003.pth',device='cuda:0'):
        self.device = device
        cnet = SVCNN('mvcnn', nclasses=40, pretraining=True, cnn_name='vgg16')
        checkpoint = torch.load(model_path)
        cnet.load_state_dict(checkpoint)
        self.model = MVCNN('mvcnn',cnet,nclasses=40,cnn_name='vgg16',num_views=1).to(device)
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

    def test(self,in_path,out_path=''):
        image = Image.open(in_path)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.unsqueeze(0)
        out = self.model(image_tensor)
        pred = torch.max(out, 1)[1]
        x = torch.max(out, 1)
        m = torch.nn.Softmax(dim=1).to(device)
        pred_score = m(out)[0,pred]
        pred_score = round(pred_score.item(), 3)
        pred_score = str(pred_score)

        pred_class = self.classnames[pred]
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        # 定义要绘制的文字内容
        text = pred_class + " " + pred_score
        # 定义文字颜色和字体
        text_color = (128, 128, 128)
        # font = ImageFont.truetype("arial.ttf", size=30)  # 使用 Arial 字体，大小为 30
        # 获取文字的宽度和高度
        text_width, text_height = draw.textsize(text)

        # 计算文字的位置（这里假设将文字放在图片中心）
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 8

        # 在图片上绘制文字
        draw.text((x, y), text, fill=text_color)

        # 保存输出的图片
        save_name = in_path.split('/')[-1].split('.')[0] + '_out.jpg'
        if out_path[-1] != "/":
            out_path = out_path + '/'
        image.save(out_path + save_name)
        # os.walk()
        # 显示图片
        # image.show()
        return pred_class
def save_results(source, destination,test):
    # 创建目标文件夹
    os.makedirs(destination, exist_ok=True)

    # 遍历源文件夹中的所有文件和文件夹
    for item in os.listdir(source):
        # 构建源文件/文件夹的完整路径
        source_path = os.path.join(source, item)
        # 构建目标文件/文件夹的完整路径
        destination_path = os.path.join(destination, item)

        # 如果是文件，则复制文件
        if os.path.isfile(source_path):
            pred = test.test(source_path,destination)
        # 如果是文件夹，则递归调用函数处理子文件夹
        elif os.path.isdir(source_path):
            save_results(source_path, destination_path,test)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_path", "--in_path", type=str, help="输入路径", default="")
    parser.add_argument("-out_path", "--out_path", type=str, help="输出路径",
                        default='./img_results')
    parser.add_argument("-device", "--device", type=str, help="使用设备,例如 cuda:0   cpu",
                        default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    in_path = args.in_path
    out_path = args.out_path
    if os.path.isdir(in_path):
        test = Test(device=device)
        save_results(in_path,out_path,test)
    elif os.path.isfile(in_path):
        test = Test()
        pred = test.test(in_path,out_path)
    else:
        print("该路径不存在")


