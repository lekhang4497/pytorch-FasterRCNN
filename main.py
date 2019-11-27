import torchvision as tv
import torch
import cv2
import numpy as np

from PennFudanDataset import PennFudanDataset
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def create_model():
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    return model


def visualize_image(image, prediction):
    boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
    image = image.transpose(1, 2, 0) * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        cv2.rectangle(image, (x0, y0), (x1, y1),
                color=(0, 255, 0), thickness=1)
        cv2.putText(image, str(labels[i].item()) + ':' + '%.2f'%(scores[i].item()), (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=1)
    cv2.imwrite('output/prediction.jpg', image)
    pass

def inference(model, dataset):
    image, target = dataset[0]
    
    if torch.cuda.is_available():
        model = model.to('cpu')
        model.eval()
        images = [image.to('cpu')]
    
    predictions = model(images)
    print(predictions)
    visualize_image(images[0].numpy(), predictions[0])
    pass
if __name__ == "__main__":
    model = create_model()
    dataset = PennFudanDataset('PennFudanPed/', get_transform(False))
    inference(model, dataset)
    pass