"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim
import math
#from model import YOLOv3
from yolo_model_with_32_WITH_FLOW_Block import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    nan_values =0
    for batch_idx, (x, y, l1 , a , b, c ) in enumerate(loop):

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        a = a.to(config.DEVICE)
        b0 , b1 , b2 = (
            b[0].to(config.DEVICE),
            b[1].to(config.DEVICE),
            b[2].to(config.DEVICE),
        )

        #print(" y0, y1, y2::", y0, y1, y2)
        with torch.cuda.amp.autocast():
            out = model([x, a])
            #print('out[0].shape:',out[0].shape)

            loss1 = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])

            )
            '''
            loss2 = (
                loss_fn(out[0], b0, scaled_anchors[0])
                + loss_fn(out[1], b1, scaled_anchors[1])
                + loss_fn(out[2], b2, scaled_anchors[2])
                
            )
            '''
            #loss = loss1 + loss2
            loss = loss1
            #print("loss1:", loss1)
            #print("loss2:" ,loss2)
            #print("loss:" , loss)
        if math.isnan(loss):
            nan_values += 1
        else:
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)
    print("Number of Nan Losses:", nan_values)


def main():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=r'C:\Users\azure.lavdierada\YoloV3' + "/train.csv", test_csv_path=r'C:\Users\azure.lavdierada\YoloV3' + "/test.csv", eval_csv_path=r'C:\Users\azure.lavdierada\YoloV3' + "/test1.csv"
    )

    print("train_loader::", len(train_loader))
    print("train_eval_loader::", len(train_eval_loader))
    print("test_loader::", len(test_loader))


    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
        plot_couple_examples(model, train_eval_loader, 0.6, 0.5, scaled_anchors)
        #print("check_class_accuracy On Test loader:")
        #check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        check_class_accuracy(model, train_eval_loader, threshold=config.CONF_THRESHOLD)
    
        pred_boxes, true_boxes = get_evaluation_bboxes(
            train_eval_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        print("Num of Pred Boxes", len(pred_boxes))
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")

        
        


    for epoch in range(config.NUM_EPOCHS):


        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL and epoch % 4 == 0:
            save_checkpoint(model, optimizer, filename=f"checkpoint1.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        
        #if epoch > 0 and epoch % 8 == 0:
        if epoch > 0 and epoch % 8 == 0:
            check_class_accuracy(model, train_eval_loader, threshold=config.CONF_THRESHOLD)
        
            pred_boxes, true_boxes = get_evaluation_bboxes(
                train_eval_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            print("Num of Pred Boxes", len(pred_boxes))
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            
            model.train()
        

def evalaute_the_model():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=r'C:\Users\azure.lavdierada\YoloV3' + "/train.csv", test_csv_path=r'C:\Users\azure.lavdierada\YoloV3' + "/test.csv", eval_csv_path=r'C:\Users\azure.lavdierada\YoloV3' + "/test1.csv"
    )

    print("train_loader::", len(train_loader))
    print("train_eval_loader::", len(train_eval_loader))
    print("test_loader::", len(test_loader))

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
        plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        print("check accuracy On Test loader:")
        check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            test_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        print("Compute MAP On Test loader:")
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")
        model.train()



if __name__ == "__main__":
    main()
    #evalaute_the_model()