import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
import datetime

from HAT_Net import HAT_Net_tiny


class Config:
    dataset_path = '/root/autodl-tmp/imagenet'
    save_path = './checkpoints'
    log_file = './checkpoints/training_log.txt'

    batch_size = 256
    accumulation_steps = 4

    epochs = 300

    lr = 1e-3

    weight_decay = 0.05

    img_size = 224

    num_classes = 1000

    warmup_epochs = 20

    use_amp = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    early_stop_patience = 20


def write_log(log_path, message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def get_data_loaders(config):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(config.dataset_path, 'train')
    val_path = os.path.join(config.dataset_path, 'val')

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("检测到预分文件夹，正在加载...")
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        test_dataset = val_dataset
    else:
        print("未检测到分层目录，正在执行代码内随机划分...")
        full_dataset = datasets.ImageFolder(config.dataset_path)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        class DatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, index):
                x, y = self.subset[index]
                if self.transform:
                    x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.subset)

        train_dataset = DatasetWrapper(train_ds, train_transform)
        val_dataset = DatasetWrapper(val_ds, val_transform)
        test_dataset = DatasetWrapper(test_ds, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=True,
                            persistent_workers=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=True,
                             persistent_workers=True, drop_last=False)

    return train_loader, val_loader, test_loader


@torch.no_grad()
def final_evaluate(model, loader, config):
    print()
    print("-" * 30)
    print("正在执行终极测试 (Final Test)...")
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc='Testing'):
        inputs, targets = inputs.to(config.device, non_blocking=True), targets.to(config.device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    final_acc = 100. * correct / total
    print(f"终极测试准确率: {final_acc:.2f}%")
    print("-" * 30)
    print()
    return final_acc


def train_one_epoch(model, loader, optimizer, criterion, config, mixup_fn, scaler, epoch, model_ema=None):
    model.train()
    running_loss = 0.0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f'Epoch [{epoch + 1}/{config.epochs}] Training', leave=False, mininterval=10.0)

    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(config.device, non_blocking=True), targets.to(config.device, non_blocking=True)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / config.accumulation_steps

        if config.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % config.accumulation_steps == 0:
            if config.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None:
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                model_ema.update(base_model)

        running_loss += loss.item() * config.accumulation_steps
        pbar.set_postfix(loss=f"{loss.item() * config.accumulation_steps:.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, config, epoch):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in tqdm(loader, desc=f'Epoch [{epoch + 1}/{config.epochs}] Validating', leave=False):
        inputs, targets = inputs.to(config.device, non_blocking=True), targets.to(config.device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return val_loss / len(loader), acc


def plot_curves(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Improvement')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_log.png')
    print("可视化结果已保存为 training_log.png，请在文件浏览器中查看。")


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    config = Config()

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
        print(f"已创建模型保存目录: {config.save_path}")

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        label_smoothing=0.1,
        num_classes=config.num_classes
    )

    model = HAT_Net_tiny(num_classes=config.num_classes).to(config.device)
    model.reset_drop_path(0.05)

    if config.device == 'cuda':
        print("正在编译模型 (torch.compile)... 首次运行可能需要几分钟，请耐心等待。")
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode='default')
        except Exception as e:
            print(f"模型编译失败，回退到普通模式: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, foreach=True)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_epochs
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_epochs]
    )

    criterion = SoftTargetCrossEntropy()

    try:
        train_loader, val_loader, test_loader = get_data_loaders(config)
    except Exception as e:
        print(f"\033[91m数据加载失败: {e}\033[0m")
        return

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    start_epoch = 0
    last_ckpt_path = os.path.join(config.save_path, 'last_checkpoint.pth')

    if os.path.exists(last_ckpt_path):
        print(f"检测到上次训练存档: {last_ckpt_path}，正在恢复...")
        checkpoint = torch.load(last_ckpt_path, map_location=config.device)

        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint.get('history', history)
        msg = f"成功恢复！将从第 {start_epoch + 1} 轮继续训练，当前最佳准确率: {best_acc:.2f}%"
        print(msg)
        write_log(config.log_file, "\n" + "=" * 50)
        write_log(config.log_file, msg)

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_ema = ModelEmaV2(base_model, decay=0.9999)

    if os.path.exists(last_ckpt_path) and 'model_ema_state_dict' in checkpoint:
        model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
        print("已从断点恢复 EMA 模型状态！")
    else:
        print("已从当前模型全新初始化 EMA 权重！")

    print(f"开始在设备 {config.device} 上执行训练任务...")
    write_log(config.log_file, f"=== 开始在 {config.device} 上执行训练任务 ===")

    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config, mixup_fn, scaler, epoch,
                                     model_ema)

        val_loss, val_acc = validate(model_ema.module, val_loader, config, epoch)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - start_time

        log_msg = f"Epoch {epoch + 1}/{config.epochs} | 耗时: {epoch_time:.1f}s | 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.2f}% (EMA)"
        print(log_msg)
        write_log(config.log_file, log_msg)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
            'model_ema_state_dict': model_ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc,
            'history': history
        }, os.path.join(config.save_path, 'last_checkpoint.pth'))

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            save_file = os.path.join(config.save_path, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': val_acc,
            }, save_file)
            best_msg = f"--> 发现更优 EMA 模型！已保存至: {save_file}"
            print(best_msg)
            write_log(config.log_file, best_msg)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stop_patience:
            stop_msg = f"早停机制触发：连续 {config.early_stop_patience} 轮性能未提升。最高准确率: {best_acc:.2f}%"
            print(stop_msg)
            write_log(config.log_file, stop_msg)
            break

    best_checkpoint = torch.load(os.path.join(config.save_path, 'best_model.pth'))
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.load_state_dict(best_checkpoint['model_state_dict'])

    final_acc = final_evaluate(base_model, test_loader, config)

    write_log(config.log_file, "=" * 50)
    write_log(config.log_file, f"终极测试准确率 (Final Test Accuracy): {final_acc:.2f}%")
    write_log(config.log_file, "=== 实验圆满结束 ===")

    plot_curves(history)


def run_dummy_test():
    config = Config()
    config.device = 'cpu'
    config.batch_size = 2
    config.epochs = 10

    print("\n" + "=" * 40)
    print("正在启动：随机噪声测试模式 (CPU专用)")
    print("目的：在没有数据集的情况下验证模型逻辑与可视化")
    print("=" * 40)

    model = HAT_Net_tiny(num_classes=config.num_classes).to(config.device)

    dummy_inputs = torch.randn(config.batch_size, 3, 224, 224).to(config.device)
    dummy_targets = torch.randint(0, 1000, (config.batch_size,)).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    model.train()
    pbar = tqdm(range(config.epochs), desc="Dummy Testing", leave=True)

    for epoch in pbar:
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        optimizer.step()

        history['train_loss'].append(loss.item())
        history['val_loss'].append(loss.item() * 1.1)
        history['val_acc'].append(min(100.0, epoch * 10.0 + 5.0))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print("\n" + "=" * 40)
    print("测试成功！正在生成可视化报告...")
    plot_curves(history)
    print("结论：HAT-Net 模型结构与可视化逻辑运行正常。")
    print("=" * 40)


if __name__ == '__main__':
    main()