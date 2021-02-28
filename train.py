import random
import os
from importlib import import_module

import numpy as np
from torch.utils.data import Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from model import *
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    seed_everything(42)

    # -- parameters
    img_root = os.getenv("IMG_ROOT")
    label_path = os.getenv("LABEL_PATH")

    model_name = "VGG19"
    use_pretrained = True
    freeze_backbone = False

    val_split = 0.4
    batch_size = 64
    num_workers = 32  # todo : fix
    num_classes = 3

    num_epochs = 100
    lr = 1e-4
    lr_classifier = 1e-3
    lr_decay_step = 10
    criterion_name = 'label_smoothing'

    train_log_interval = 20
    name = "04_lr_targeting"

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- model
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes,
        pretrained=use_pretrained,
        freeze=freeze_backbone
    ).to(device)

    # -- data_loader
    dataset = MaskBaseDataset(img_root, label_path, 'train')
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    val_set.dataset.set_phase("test")  # todo : fix

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    # -- loss & metric
    criterion = create_criterion(criterion_name)
    optimizer = Adam([
        {'params': filter(lambda p: p.requires_grad, model.backbone_layers().parameters())},
        {'params': filter(lambda p: p.requires_grad, model.classifier_layers().parameters()), 'lr': lr_classifier}
    ], lr=lr, weight_decay=5e-4)

    scheduler = StepLR(optimizer, lr_decay_step, gamma=0.5)
    # metrics = []
    # callbacks = []

    # -- logging
    logger = SummaryWriter(log_dir=f"results/{name}")

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % train_log_interval == 0:
                train_loss = loss_value / train_log_interval
                train_acc = matches / batch_size / train_log_interval
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch[{epoch}/{num_epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            if val_loss < best_val_loss:
                print("New best model for val loss! saving the model..")
                torch.save(model.state_dict(), f"results/{name}/{epoch:03}_loss_{val_loss:4.2}.ckpt")
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model..")
                torch.save(model.state_dict(), f"results/{name}/{epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
                best_val_acc = val_acc
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            print()
