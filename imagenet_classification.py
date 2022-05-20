import os
import PIL
from PIL import Image
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms


def load_model():
    model_folder = os.path.dirname(__file__)
    model_file  = os.path.join(model_folder, 'mobilenet_v3_small-047dcff4.pth')
    class_file  = os.path.join(model_folder, 'imagenet_classes.txt')
    
    model = models.mobilenet_v3_small()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    with open(class_file, 'r') as f:
        classes = [s.strip() for s in f]

    return model, classes


def run_inference(image, model, classes, limit):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )])
    
    input_t = transform(image).unsqueeze(0)
    out_t = torch.nn.functional.softmax(model(input_t), dim=1)
    scores, indices = out_t.flatten().topk(limit)
    scores = scores.detach().tolist()
    indices = indices.detach().tolist()
    class_names = [classes[idx] for idx in indices]

    return class_names, scores


if __name__ == '__main__':
    model, classes = load_model()

    image_file = 'test_image.jpg'
    image = Image.open(image_file)

    limit = 10
    class_names, scores = run_inference(image, model, classes, limit)
    for i in range(limit):
        print(f"{i}: {class_names[i]}, score={scores[i]}")

