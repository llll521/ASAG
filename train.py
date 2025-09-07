# coding=gbk
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import copy
import math
import dataloader
import model as md
import numpy as np



torch.cuda.empty_cache()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
set_seed()


def multi_task_loss(
        reg_output,
        cls_output,
        reg_target,
        cls_target,
        current_step: int,  # 当前训练步数（或epoch数）
        total_steps: int,  # 总训练步数（或总epoch数）
        initial_cls_weight: float = 0.1,  # 初始分类权重（建议接近1）
        final_cls_weight: float = 0.9,  # 最终分类权重
        initial_reg_weight: float = 0.1,  # 初始回归权重
        final_reg_weight: float = 0.9,  # 最终回归权重
        schedule: str = 'linear'  # 调整策略：linear/cosine/exponential
):
    """
    动态调整多任务损失权重

    参数:
        reg_output: 回归模型输出，形状 [batch_size, 1]
        cls_output: 分类模型输出，形状 [batch_size, num_classes]
        reg_target: 回归目标值，形状 [batch_size, 1]
        cls_target: 分类目标标签，形状 [batch_size]
        current_step: 当前训练步数或epoch数
        total_steps: 总训练步数或总epoch数
        initial_cls_weight: 初始分类损失权重（默认0.9）
        final_cls_weight: 最终分类损失权重（默认0.1）
        initial_reg_weight: 初始回归损失权重（默认0.9）
        final_reg_weight: 最终回归损失权重（默认0.1）
        schedule: 权重调整策略，可选 linear/cosine/exponential（默认linear）

    返回:
        total_loss: 加权后的总损失
    """
    # 确保训练进度比例在[0,1]之间
    progress = min(max(current_step / total_steps, 0.0), 1.0)

    # 根据策略计算分类权重衰减系数
    if schedule == 'linear':
        cls_weight = initial_cls_weight - (initial_cls_weight - final_cls_weight) * progress
        reg_weight = initial_reg_weight + (final_reg_weight - initial_reg_weight) * progress
    elif schedule == 'cosine':
        # 余弦退火式调整
        cls_weight = final_cls_weight + 0.5 * (initial_cls_weight - final_cls_weight) * (
                1 + math.cos(math.pi * progress))
        reg_weight = final_reg_weight - 0.5 * (final_reg_weight - initial_reg_weight) * (
                1 + math.cos(math.pi * progress))
    elif schedule == 'exponential':
        # 指数衰减
        decay_rate = 10.0  # 控制衰减速度
        cls_weight = final_cls_weight + (initial_cls_weight - final_cls_weight) * math.exp(-decay_rate * progress)
        reg_weight = initial_reg_weight + (final_reg_weight - initial_reg_weight) * (
                1 - math.exp(-decay_rate * progress))
    else:
        raise ValueError(f"Unsupported schedule: {schedule}, choose from ['linear', 'cosine', 'exponential']")

    # 计算损失
    reg_loss = F.mse_loss(reg_output, reg_target.unsqueeze(1))
    cls_loss = F.cross_entropy(cls_output, cls_target)

    # 动态加权
    total_loss = reg_weight * reg_loss + cls_weight * cls_loss

    return total_loss


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    all_reg_preds = []
    all_reg_labels = []
    all_cls_preds = []
    all_cls_labels = []

    for encoding_q, encoding_a, reg_labels, cls_labels in tqdm(dataloader, desc="Training"):
        encoding_q = {k: v.to(device) for k, v in encoding_q.items()}
        encoding_a = {k: v.to(device) for k, v in encoding_a.items()}
        reg_labels = reg_labels.float().to(device)
        cls_labels = cls_labels.long().to(device)

        optimizer.zero_grad()

        # 前向传播
        reg_output, cls_output = model(encoding_q, encoding_a)

        # 计算损失
        loss = multi_task_loss(
            reg_output=reg_output,
            cls_output=cls_output,
            reg_target=reg_labels,
            cls_target=cls_labels,
            current_step=epoch,
            total_steps=50,
            initial_cls_weight=0.9,
            final_cls_weight=0.1,
            initial_reg_weight=0.9,
            final_reg_weight=0.1,
            schedule='cosine'
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        total_loss += loss.item()

        # 收集预测结果
        all_reg_preds.extend(reg_output.detach().cpu().numpy())
        all_reg_labels.extend(reg_labels.cpu().numpy())
        all_cls_preds.extend(torch.argmax(cls_output, dim=1).detach().cpu().numpy())

        all_cls_labels.extend(cls_labels.cpu().numpy())

    # 计算指标
    avg_total_loss = total_loss / len(dataloader)

    # 回归指标
    reg_rmse = root_mean_squared_error(all_reg_labels, all_reg_preds)
    #
    #
    all_reg_labels = np.array(all_reg_labels, dtype=float)
    all_reg_preds = np.array(all_reg_preds, dtype=float)

    # 处理缺失值
    all_reg_labels = np.nan_to_num(all_reg_labels)
    all_reg_preds = np.nan_to_num(all_reg_preds)
    all_reg_labels = all_reg_labels.flatten()
    all_reg_preds = all_reg_preds.flatten()

    reg_pearson, _ = pearsonr(all_reg_labels, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_labels, all_reg_preds)

    all_reg_preds_discrete = np.round(all_reg_preds).astype(int)
    all_reg_labels_discrete = np.round(all_reg_labels).astype(int)

    # 分类指标
    cls_accuracy = accuracy_score(all_cls_labels, all_cls_preds)
    cls_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
    cls_recall = recall_score(all_cls_labels, all_cls_preds, average='macro')

    return (avg_total_loss,
            reg_pearson, reg_rmse, reg_mae, cls_accuracy, cls_f1, cls_recall)


def evaluate_model(model, dataloader, device, epoch, total_epochs):
    model.eval()
    total_loss = 0.0

    all_reg_preds = []
    all_reg_labels = []
    all_cls_preds = []
    all_cls_labels = []

    with torch.no_grad():
        for encoding_q, encoding_a, reg_labels, cls_labels in tqdm(dataloader, desc="Evaluating"):
            encoding_q = {k: v.to(device) for k, v in encoding_q.items()}
            encoding_a = {k: v.to(device) for k, v in encoding_a.items()}
            reg_labels = reg_labels.float().to(device)
            cls_labels = cls_labels.long().to(device)

            # 前向传播
            reg_output, cls_output = model(encoding_q, encoding_a)
            # 计算损失

            loss = multi_task_loss(
                reg_output=reg_output,
                cls_output=cls_output,
                reg_target=reg_labels,
                cls_target=cls_labels,
                current_step=epoch,
                total_steps=50,
                initial_cls_weight=0.9,
                final_cls_weight=0.1,
                initial_reg_weight=0.9,
                final_reg_weight=0.1,
                schedule='cosine'
            )

            # 记录损失
            total_loss += loss.item()

            # 收集预测结果
            all_reg_preds.extend(reg_output.cpu().numpy())
            all_reg_labels.extend(reg_labels.cpu().numpy())
            all_cls_preds.extend(torch.argmax(cls_output, dim=1).cpu().numpy())
            all_cls_labels.extend(cls_labels.view(-1).cpu().numpy())

    # 计算指标
    avg_total_loss = total_loss / len(dataloader)

    # 回归指标
    reg_rmse = root_mean_squared_error(all_reg_labels, all_reg_preds)
    #
    all_reg_labels = np.array(all_reg_labels, dtype=float)
    all_reg_preds = np.array(all_reg_preds, dtype=float)

    # 处理缺失值
    all_reg_labels = np.nan_to_num(all_reg_labels)
    all_reg_preds = np.nan_to_num(all_reg_preds)
    all_reg_labels = all_reg_labels.flatten()
    all_reg_preds = all_reg_preds.flatten()

    reg_pearson, _ = pearsonr(all_reg_labels, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_labels, all_reg_preds)  # 新增

    all_reg_preds_discrete = np.round(all_reg_preds).astype(int)
    all_reg_labels_discrete = np.round(all_reg_labels).astype(int)

    # 分类指标
    cls_accuracy = accuracy_score(all_cls_labels, all_cls_preds)
    cls_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
    cls_recall = recall_score(all_cls_labels, all_cls_preds, average='macro')

    return (avg_total_loss,
            reg_pearson, reg_rmse, reg_mae, cls_accuracy, cls_f1, cls_recall)


#
def cross_validation(
        model_save_path,
        train_data_path,  # 训练集路径
        val_data_path,  # 新验证集路径
        test_data_path,  # 测试集路径
        num_epochs=50,
        batch_size=25,
        learning_rate=2e-5,
        random_state=20
):
    # ===== 加载预划分的数据集 =====
    # 加载训练集
    train_dataset = dataloader.QADataset(
        r"train.txt",  # 使用预定义的训练集路径

        tokenizer_name="bert-base",
        max_length=128
    )
    # 加载验证集
    val_dataset = dataloader.QADataset(
        r"val.txt",  # 使用预定义的验证集路径

        tokenizer_name="bert-base",
        max_length=128
    )
    # 加载测试集
    test_dataset = dataloader.QADataset(
        r"test.txt",  # 使用预定义的测试集路径

        tokenizer_name="bert-base",
        max_length=128
    )

    # 检查数据集是否为空
    for name, dataset in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        if len(dataset) == 0:
            raise ValueError(f"{name}为空，请检查数据路径！")
    print(f"训练样本数: {len(train_dataset)} | 验证样本数: {len(val_dataset)} | 测试样本数: {len(test_dataset)}")

    # ===== 创建DataLoader =====
    # 注意：验证集和测试集不需要shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 仅训练集需要shuffle
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集关闭shuffle
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集关闭shuffle
        drop_last=True
    )

    # ===== 设备设置 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 模型初始化 =====
    subnet = md.Subnet()
    model = md.SNN(subnet)
    model.to(device)  # 使用动态设备分配
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # 周期长度
        eta_min=1e-6  # 最小学习率
    )

    # ===== 结果记录 =====
    train_results = {
        'total_loss': [], 'lr': [],
        'reg_pearson': [], 'reg_rmse': [], 'reg_mae': [],
        'cls_accuracy': [], 'cls_f1': [], 'cls_recall': []
    }
    val_results = {
        'total_loss': [], 'lr': [],
        'reg_pearson': [], 'reg_rmse': [], 'reg_mae': [],
        'cls_accuracy': [], 'cls_f1': [], 'cls_recall': []
    }
    best_val_rmse = float('inf')
    best_model = None

    # ===== 训练循环 =====
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        # 训练阶段
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        train_loss, train_pearson, train_rmse, train_mae, train_acc, train_f1, train_recall = train_metrics

        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, device, epoch, num_epochs)
        val_loss, val_pearson, val_rmse, val_mae, val_acc, val_f1, val_recall = val_metrics

        # 每个epoch结束后调用（不需要指标）：
        scheduler.step()
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 训练指标
        train_results['total_loss'].append(train_loss)
        train_results['reg_pearson'].append(train_pearson)
        train_results['reg_rmse'].append(train_rmse)
        train_results['reg_mae'].append(train_mae)
        train_results['lr'].append(current_lr)
        train_results['cls_accuracy'].append(train_acc)
        train_results['cls_f1'].append(train_f1)
        train_results['cls_recall'].append(train_recall)

        # 验证指标
        val_results['total_loss'].append(val_loss)
        val_results['reg_pearson'].append(val_pearson)
        val_results['reg_rmse'].append(val_rmse)
        val_results['reg_mae'].append(val_mae)
        val_results['lr'].append(current_lr)
        val_results['cls_accuracy'].append(val_acc)
        val_results['cls_f1'].append(val_f1)
        val_results['cls_recall'].append(val_recall)

        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(model_save_path, "best_model.pt"))
            print(f"最佳模型已保存 (验证集rmse: {val_rmse:.4f})")

        # 打印进度
        print(f"[训练集] 损失: {train_loss:.4f} | acc: {train_rmse:.4f}")
        print(f"[验证集] 损失: {val_loss:.4f} | acc: {val_rmse:.4f}")

    # ===== 保存训练结果 =====
    results_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        **{f'train_{k}': v for k, v in train_results.items()},
        **{f'val_{k}': v for k, v in val_results.items()}
    })
    results_df.to_csv(os.path.join(model_save_path, "training_results4_reg_cls_rmse_model1.csv"), index=False)

    # ===== 最终测试 =====
    print("\n===== 在测试集上最终评估 =====")
    model.load_state_dict(
        torch.load(os.path.join(model_save_path, "best_model.pt"), map_location=device, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device, epoch, num_epochs)
    test_loss, test_pearson, test_rmse, test_mae, test_acc, test_f1, test_recall = test_metrics

    # 打印测试结果
    print(f"[测试集] 总损失: {test_loss:.4f}")
    print(
        f"        Pearson系数: {test_pearson:.4f} | rMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} ")  # 更新输出
    print(f"        precision精确率: {test_acc:.4f} | F1分数: {test_f1:.4f} | 召回率: {test_recall:.4f}")

    # 保存最终测试结果
    test_results = {
        'test_total_loss': [test_loss],
        'test_reg_pearson': [test_pearson],
        'test_reg_rmse': [test_rmse],
        'test_reg_mae': [test_mae],
        'test_cls_accuracy': [test_acc],
        'test_cls_f1': [test_f1],
        'test_cls_recall': [test_recall]
    }
    pd.DataFrame(test_results).to_csv(os.path.join(model_save_path, "test_results4_reg_cls_rmse_model1.csv"),
                                      index=False)


if __name__ == "__main__":
    model_save_directory = "./multi_task_models_zh"
    os.makedirs(model_save_directory, exist_ok=True)

    # 假设预划分数据集路径如下，根据实际情况修改
    cross_validation(
        model_save_path=model_save_directory,
        train_data_path=r"train.txt",  # 预划分训练集
        val_data_path=r"val.txt",  # 预划分验证集
        test_data_path=r"test.txt",  # 预划分测试集
        num_epochs=50,
        batch_size=25,
        learning_rate=2e-5,
        random_state=42
    )

