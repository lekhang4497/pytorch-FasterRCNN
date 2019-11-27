import torchvision as tv
import torch
import cv2
import numpy as np
import random

from PennFudanDataset import PennFudanDataset
import transforms as T
import utils
from engine import train_one_epoch, evaluate

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
    colors = [tuple([cc.item() for cc in np.random.choice(range(256), size=3)])
                            for i in range(len(boxes))]
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            x0, y0, x1, y1 = boxes[i]
            cv2.rectangle(image, (x0, y0), (x1, y1),
                    color=colors[i], thickness=2)
            cv2.putText(image, str(labels[i].item()) + ':' + '%.2f'%(scores[i].item()), (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], thickness=2)
    cv2.imwrite('output/prediction.jpg', image)
    pass

def inference(model, dataset):
    idx = random.randint(0, len(dataset))
    print(idx)
    image, target = dataset[idx]
    
    if torch.cuda.is_available():
        model = model.to('cpu')
        model.eval()
        images = [image.to('cpu')]
    
    predictions = model(images)
    print(predictions)
    visualize_image(images[0].numpy(), predictions[0])
    pass

def train(model, dataset, dataset_test):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # save model
    torch.save(model.state_dict(), 'fasterrcnn_state_dict.pth')
    print("That's it!")

    pass

if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load('fasterrcnn_state_dict.pth'))
    dataset = PennFudanDataset('PennFudanPed/', get_transform(True))
    dataset_test = PennFudanDataset('PennFudanPed/', get_transform(False))
    inference(model, dataset_test)
    #train(model, dataset, dataset_test)
    pass