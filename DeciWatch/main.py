import torch
from model.vision_transformer import ViT
from model.masked_autoencoder import MAE
import yaml
import logging
from utils.general import colorstr
from dataset import VitDataset, get_dataloader
from torchvision.transforms import transforms
import os
from tensorboardX import SummaryWriter


def vit_model():
    vit = ViT(image_size=256, patch_size=32, in_channel=3, num_classes=1000,
              embed_dim=1024, depth=6, num_heads=8, mlp_ratio=2)
    return vit


def mae_model():
    mae = MAE(
        image_size=256, patch_size=32, in_channel=3, encoder_dim=1024,
        encoder_depth=6, encoder_heads=8, drop_rate=0,      # vit encoder
        masking_ratio=0.75,   # the paper recommended 75% masked patches
        decoder_dim=512,      # paper showed good results with just 512
        decoder_depth=6       # anywhere from 1 to 8
    )
    return mae


def train(model, train_loader, epoch, loss_fn, optimizer, device, hyp_config):
    # for p in model.parameters():
    #     p.requires_grad = True
    model.train()
    loss = 0.0
    for i_batch, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        preds = model(image)
        cost = loss_fn(preds, target)
        model.zero_grad()
        cost.backward()
        optimizer.step()
        loss += cost.item()
        if (i_batch + 1) % hyp_config["display_interval"] == 0:
            logger.info("[epoch: {0}/{1}] [batch: {2}/{3}] loss: {4}".format(
                epoch, hyp_config["total_epochs"], i_batch + 1, len(train_loader), cost.item()
            ))
    train_loss = loss / (i_batch + 1)
    logger.info('Train loss: %f' % (train_loss))
    return loss


def val(model, val_loader, epoch, loss_fn, hyp_config):
    print('Start val')
    # for p in model.parameters():
    #     p.requires_grad = False
    model.eval()
    n_correct = 0
    loss = 0.0
    for i_batch, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)
        preds = model(image)
        cost = loss_fn(preds, target)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        for pred, target in zip(preds, target):
            if pred == target:
                n_correct += 1
        loss += cost.item()
        if (i_batch + 1) % hyp_config["display_interval"] == 0:
            logger.info("[epoch: {0}/{1}] [batch: {2}/{3}] loss: {4}".format(
                epoch, hyp_config["total_epochs"], i_batch + 1, len(val_loader), cost.item()
            ))

    #     if i_batch == max_i:
    #         break
    # if i_batch == max_i:
    #     total_num = max_i * hyp_config["batch_size"]
    # else:
    #     total_num = len(val_loader.dataset)

    total_num = len(val_loader.dataset)
    accuracy = n_correct / float(total_num)
    val_loss = loss / (i_batch + 1)
    logger.info('Test loss: %f, accuray: %f' % (val_loss, accuracy))

    return accuracy, loss


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    images = torch.randn(8, 3, 256, 256)

    # 1.超参数读取
    hyp_file = r"./config/hyp_scratch.yaml"
    with open(hyp_file, "r", encoding="utf-8") as f:
        hyp_config = yaml.load(f, Loader=yaml.SafeLoader)
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp_config .items()))

    # 2.数据集加载
    img_dir = r""
    train_file = r""
    val_file = r""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(hyp_config["input_size"], scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = VitDataset(img_dir, train_file, train_transform)
    val_dataset = VitDataset(img_dir, val_file, val_transform)
    train_loader = get_dataloader(train_dataset, hyp_config["batch_size"], num_workers=4, shuffle=True)
    val_loader = get_dataloader(val_dataset, hyp_config["batch_size"], num_workers=4, shuffle=True)

    # 3.模型
    device = torch.device(args.device)
    model = vit_model()
    model.to(device)
    if hyp_config["param_file"]:
        ckpt = torch.load(hyp_config["param_file"], map_location=device)  # load checkpoint
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        model.load_state_dict(state_dict)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 4. 损失函数和优化器设置
    if hyp_config['optim'] == 'adam':
        # optimizer = torch.optim.AdamW()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp_config['lr'], betas=(hyp_config['momentum'], 0.95),
                                     weight_decay=hyp_config['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp_config['lr'], momentum=hyp_config['momentum'],
                                    weight_decay=hyp_config['weight_decay'], nesterov=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5.训练和保存权重，记录日志
    writer = SummaryWriter(hyp_config['exp'])
    start_epoch, total_epochs = hyp_config["start_epoch"], hyp_config["total_epochs"]
    model_save_dir = hyp_config['model_save_dir']
    best_accuracy = 0.0
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = train(model, train_loader, epoch, loss_fn, optimizer, hyp_config)
        accuracy, val_loss = val(model, val_loader, epoch, loss_fn, hyp_config)
        print("is best accuracy: {0}".format(accuracy > best_accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_save_dir, "model_{}_{:.4f}.pth".format(epoch, accuracy)))
        elif epoch % hyp_config['save_interval'] == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, "model_{}_{:.4f}.pth".format(epoch, accuracy)))

        epoch += 1
        writer.add_scalar("TrainLoss", epoch_loss, global_step=epoch)
        writer.add_scalar("accuracy", accuracy, global_step=epoch)
        writer.add_scalar("ValLoss", val_loss, global_step=epoch)
    writer.close()

    # import timm
    # vit = timm.create_model('vit_base_patch16_224_miil', pretrained=True)
    # pred = vit(images)
    # print(pred.shape)


