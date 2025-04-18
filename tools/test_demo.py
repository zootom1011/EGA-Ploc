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

label = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
             'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']


def main():
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    if args.dataset == 'IHC':
        cfg = vislocas_get_cfg()
    else:
        cfg = hpa18_get_cfg()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for classifier_model in cfg.TRAIN.CLASSIFIER_NAME:

        for database in cfg.DATA.DATASET_NAME:

            result_prefix = "{}/results/{}".format(cfg.DATA.RESULT_DIR, database)
            log_prefix = "{}/independent".format(database)
            print(log_prefix)

            test_picture_path = args.single_image_path  # replace the path

            # Classifier
            model = getClassifier(cfg, model_name=classifier_model)
            model = model.to(device)


            # load model weights
            load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)

            model.eval()

            img = Image.open(test_picture_path).convert('RGB')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.sigmoid(output)[0]
            pred = pred.cpu().detach().numpy()

            confidences = {name: float(f"{prob:.4f}") for name, prob in zip(label, pred)}
            sorted_conf = sorted(confidences.items(), key=lambda x: -x[1])

            result_md = "**Protein may located in:**\n\n" + "\n".join(
                [f"{name}: {prob * 100:.2f}%"
                 for name, prob in sorted_conf if prob > 0.1])
            print(result_md)

            if du.is_master_proc():
                print("Test finished")



if __name__ == "__main__":
    main()