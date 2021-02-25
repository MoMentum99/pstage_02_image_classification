import random
import os

import numpy as np
from torch.utils.data import Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, MaskMultiLabelDataset
from model import *


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

    val_split = 0.1
    batch_size = 64
    num_workers = 32  # todo : fix
    num_classes = 8  # 3(mask) + 2(gender) + 3(age group)

    num_epochs = 100
    lr = 1e-4
    lr_decay_step = 10

    train_log_interval = 20
    name = "02_vgg"

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- model
    if False:
        model = AlexNet(num_classes=num_classes).to(device)
    else:
        model = VGG19(num_classes=num_classes, pretrained=True, freeze=False).to(device)

    # -- data_loader
    dataset = MaskMultiLabelDataset(img_root, label_path, 'train')
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
    criterion = nn.CrossEntropyLoss
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)
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
        mask_matches = 0
        gender_matches = 0
        age_matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, mask_labels, gender_labels, age_labels = train_batch
            inputs = inputs.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            mask_logits, gender_logits, age_logits = torch.split(outs, [3, 2, 3], dim=1)

            mask_loss = criterion(reduction='mean')(mask_logits, mask_labels)
            gender_loss = criterion(reduction='mean')(gender_logits, gender_labels)  # todo : fix to bce?
            age_loss = criterion(reduction='mean')(age_logits, age_labels)
            loss = mask_loss + gender_loss + age_loss

            mask_preds = torch.argmax(mask_logits, dim=-1)
            gender_preds = torch.argmax(gender_logits, dim=-1)
            age_preds = torch.argmax(age_logits, dim=-1)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            mask_matches += (mask_preds == mask_labels).sum().item()
            gender_matches += (gender_preds == gender_labels).sum().item()
            age_matches += (age_preds == age_labels).sum().item()
            if (idx + 1) % train_log_interval == 0:
                train_loss = loss_value / train_log_interval
                mask_acc = mask_matches / batch_size / train_log_interval
                gender_acc = gender_matches / batch_size / train_log_interval
                age_acc = age_matches / batch_size / train_log_interval
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch[{epoch}/{num_epochs}]({idx + 1}/{len(train_loader)})\n"
                    f"Loss: total {train_loss:4.4} || mask {mask_loss:4.4} || gender {gender_loss:4.4} || age {age_loss:4.4}\n"
                    f"Acc: mask {mask_acc:4.2%} || gender {gender_acc:4.2%} || age {age_acc:4.2%}"
                )
                logger.add_scalar("Train/Loss/total", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/Loss/mask", mask_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/Loss/gender", gender_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/Loss/age", age_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/Acc/mask", mask_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/Acc/gender", gender_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/Acc/age", age_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                mask_matches = 0
                gender_matches = 0
                age_matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_mask_acc_items = []
            val_gender_acc_items = []
            val_age_acc_items = []
            for val_batch in val_loader:
                inputs, mask_labels, gender_labels, age_labels = val_batch
                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                outs = model(inputs)
                mask_logits, gender_logits, age_logits = torch.split(outs, [3, 2, 3], dim=1)

                mask_preds = torch.argmax(mask_logits, dim=-1)
                gender_preds = torch.argmax(gender_logits, dim=-1)
                age_preds = torch.argmax(age_logits, dim=-1)

                mask_loss = criterion(reduction='sum')(mask_logits, mask_labels).item()
                gender_loss = criterion(reduction='sum')(gender_logits, gender_labels).item()  # todo : fix to bce?
                age_loss = criterion(reduction='sum')(age_logits, age_labels).item()
                loss_item = mask_loss + gender_loss + age_loss

                mask_matches = (mask_preds == mask_labels).sum().item()
                gender_matches = (gender_preds == gender_labels).sum().item()
                age_matches = (age_preds == age_labels).sum().item()

                val_loss_items.append(loss_item)
                val_mask_acc_items.append(mask_matches)
                val_gender_acc_items.append(gender_matches)
                val_age_acc_items.append(age_matches)

            val_loss = np.sum(val_loss_items) / len(val_set)
            val_mask_acc = np.sum(val_mask_acc_items) / len(val_set)
            val_gender_acc = np.sum(val_gender_acc_items) / len(val_set)
            val_age_acc = np.sum(val_age_acc_items) / len(val_set)
            val_acc = (val_mask_acc + val_age_acc + val_age_acc) / 3
            if val_loss < best_val_loss:
                print("New best model for val loss! saving the model..")
                torch.save(model.state_dict(), f"results/{name}/{epoch:03}_loss_{val_loss:4.2}.ckpt")
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model..")
                torch.save(model.state_dict(), f"results/{name}/{epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
                best_val_acc = val_acc
            print(
                f"[Val] mask acc : {val_mask_acc:4.2%}, gender acc : {val_gender_acc:4.2%}, age acc : {val_age_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            print()
