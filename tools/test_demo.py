import os
import sys
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from torchvision import transforms
import torch
import utils.distributed as du
from models.classifier_model import getClassifier
from models.train_classifier import load_best_classifier_model
from utils.args import parse_args
from utils.Vislocas_config import get_cfg as vislocas_get_cfg
from utils.HPA18_config import get_cfg as hpa18_get_cfg

# 定义标签列表，对应10种亚细胞结构名称
# cytoplasm：细胞质
# cytoskeleton: 细胞骨架
# endoplasmic reticulum: 内质网
# golgi apparatus: 高尔基体
# lysosomes: 溶酶体
# mitochondria: 线粒体
# nucleoli: 核仁
# nucleus: 细胞核
# plasma membrane: 质膜
# visicles: 囊泡
label = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
             'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']

def main():
    """
    Main function to spawn the test process.
    """
    # 解析命令行参数 转到 utils.args.py
    args = parse_args()

    # 根据数据集类型选择对应的配置文件
    if args.dataset == 'IHC':
        cfg = vislocas_get_cfg() # 获取 vislocas 数据集的配置文件
    else:
        cfg = hpa18_get_cfg()

    # 设置PyTorch的cuDNN后端：启用、确定性模式、关闭benchmark
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置运行设备：优先使用GPU，否则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 遍历所有待测试的分类器模型名称
    for classifier_model in cfg.TRAIN.CLASSIFIER_NAME:

        # 遍历所有待测试的数据库名称
        for database in cfg.DATA.DATASET_NAME:

            # 构造结果保存路径前缀
            result_prefix = "{}/results/{}".format(cfg.DATA.RESULT_DIR, database)
            # 构造日志路径前缀
            log_prefix = "{}/independent".format(database)
            # 打印当前数据库日志前缀
            print(log_prefix)

            # 获取单张测试图片路径（由命令行参数传入）
            test_picture_path = args.single_image_path  # replace the path

            # 初始化分类器模型
            model = getClassifier(cfg, model_name=classifier_model)
            # 将模型迁移到指定设备（GPU/CPU）
            model = model.to(device)

            # 加载训练好的最优模型权重
            load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)

            # 切换模型到评估模式（关闭dropout等训练层）
            model.eval()

            # 读取并转换测试图片为RGB格式
            img = Image.open(test_picture_path).convert('RGB')
            # 定义图像预处理：转为Tensor并做ImageNet标准化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # 对图像进行预处理并增加batch维度，再送到设备
            img_tensor = transform(img).unsqueeze(0).to(device)

            # 关闭梯度计算，进行前向推理
            with torch.no_grad():
                output = model(img_tensor)
                # 使用sigmoid将输出转为概率，取第0个样本
                pred = torch.sigmoid(output)[0]
            # 将预测概率转移到CPU并转为numpy数组
            pred = pred.cpu().detach().numpy()

            # 将标签与对应概率组合成字典，保留4位小数
            confidences = {name: float(f"{prob:.4f}") for name, prob in zip(label, pred)}
            # 按概率降序排序
            sorted_conf = sorted(confidences.items(), key=lambda x: -x[1])

            # 构造Markdown格式的结果字符串：仅保留概率>0.1的项，乘以100转为百分比
            result_md = "**Protein may located in:**\n\n" + "\n".join(
                [f"{name}: {prob * 100:.2f}%"
                 for name, prob in sorted_conf if prob > 0.1])
            # 打印预测结果
            print(result_md)

            # 如果是主进程，打印测试结束提示
            if du.is_master_proc():
                print("Test finished")

if __name__ == "__main__":
    main()