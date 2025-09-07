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


# �����������
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
        current_step: int,  # ��ǰѵ����������epoch����
        total_steps: int,  # ��ѵ������������epoch����
        initial_cls_weight: float = 0.1,  # ��ʼ����Ȩ�أ�����ӽ�1��
        final_cls_weight: float = 0.9,  # ���շ���Ȩ��
        initial_reg_weight: float = 0.1,  # ��ʼ�ع�Ȩ��
        final_reg_weight: float = 0.9,  # ���ջع�Ȩ��
        schedule: str = 'linear'  # �������ԣ�linear/cosine/exponential
):
    """
    ��̬������������ʧȨ��

    ����:
        reg_output: �ع�ģ���������״ [batch_size, 1]
        cls_output: ����ģ���������״ [batch_size, num_classes]
        reg_target: �ع�Ŀ��ֵ����״ [batch_size, 1]
        cls_target: ����Ŀ���ǩ����״ [batch_size]
        current_step: ��ǰѵ��������epoch��
        total_steps: ��ѵ����������epoch��
        initial_cls_weight: ��ʼ������ʧȨ�أ�Ĭ��0.9��
        final_cls_weight: ���շ�����ʧȨ�أ�Ĭ��0.1��
        initial_reg_weight: ��ʼ�ع���ʧȨ�أ�Ĭ��0.9��
        final_reg_weight: ���ջع���ʧȨ�أ�Ĭ��0.1��
        schedule: Ȩ�ص������ԣ���ѡ linear/cosine/exponential��Ĭ��linear��

    ����:
        total_loss: ��Ȩ�������ʧ
    """
    # ȷ��ѵ�����ȱ�����[0,1]֮��
    progress = min(max(current_step / total_steps, 0.0), 1.0)

    # ���ݲ��Լ������Ȩ��˥��ϵ��
    if schedule == 'linear':
        cls_weight = initial_cls_weight - (initial_cls_weight - final_cls_weight) * progress
        reg_weight = initial_reg_weight + (final_reg_weight - initial_reg_weight) * progress
    elif schedule == 'cosine':
        # �����˻�ʽ����
        cls_weight = final_cls_weight + 0.5 * (initial_cls_weight - final_cls_weight) * (
                1 + math.cos(math.pi * progress))
        reg_weight = final_reg_weight - 0.5 * (final_reg_weight - initial_reg_weight) * (
                1 + math.cos(math.pi * progress))
    elif schedule == 'exponential':
        # ָ��˥��
        decay_rate = 10.0  # ����˥���ٶ�
        cls_weight = final_cls_weight + (initial_cls_weight - final_cls_weight) * math.exp(-decay_rate * progress)
        reg_weight = initial_reg_weight + (final_reg_weight - initial_reg_weight) * (
                1 - math.exp(-decay_rate * progress))
    else:
        raise ValueError(f"Unsupported schedule: {schedule}, choose from ['linear', 'cosine', 'exponential']")

    # ������ʧ
    reg_loss = F.mse_loss(reg_output, reg_target.unsqueeze(1))
    cls_loss = F.cross_entropy(cls_output, cls_target)

    # ��̬��Ȩ
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

        # ǰ�򴫲�
        reg_output, cls_output = model(encoding_q, encoding_a)

        # ������ʧ
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

        # ���򴫲�
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ��¼��ʧ
        total_loss += loss.item()

        # �ռ�Ԥ����
        all_reg_preds.extend(reg_output.detach().cpu().numpy())
        all_reg_labels.extend(reg_labels.cpu().numpy())
        all_cls_preds.extend(torch.argmax(cls_output, dim=1).detach().cpu().numpy())

        all_cls_labels.extend(cls_labels.cpu().numpy())

    # ����ָ��
    avg_total_loss = total_loss / len(dataloader)

    # �ع�ָ��
    reg_rmse = root_mean_squared_error(all_reg_labels, all_reg_preds)
    #
    #
    all_reg_labels = np.array(all_reg_labels, dtype=float)
    all_reg_preds = np.array(all_reg_preds, dtype=float)

    # ����ȱʧֵ
    all_reg_labels = np.nan_to_num(all_reg_labels)
    all_reg_preds = np.nan_to_num(all_reg_preds)
    all_reg_labels = all_reg_labels.flatten()
    all_reg_preds = all_reg_preds.flatten()

    reg_pearson, _ = pearsonr(all_reg_labels, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_labels, all_reg_preds)

    all_reg_preds_discrete = np.round(all_reg_preds).astype(int)
    all_reg_labels_discrete = np.round(all_reg_labels).astype(int)

    # ����ָ��
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

            # ǰ�򴫲�
            reg_output, cls_output = model(encoding_q, encoding_a)
            # ������ʧ

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

            # ��¼��ʧ
            total_loss += loss.item()

            # �ռ�Ԥ����
            all_reg_preds.extend(reg_output.cpu().numpy())
            all_reg_labels.extend(reg_labels.cpu().numpy())
            all_cls_preds.extend(torch.argmax(cls_output, dim=1).cpu().numpy())
            all_cls_labels.extend(cls_labels.view(-1).cpu().numpy())

    # ����ָ��
    avg_total_loss = total_loss / len(dataloader)

    # �ع�ָ��
    reg_rmse = root_mean_squared_error(all_reg_labels, all_reg_preds)
    #
    all_reg_labels = np.array(all_reg_labels, dtype=float)
    all_reg_preds = np.array(all_reg_preds, dtype=float)

    # ����ȱʧֵ
    all_reg_labels = np.nan_to_num(all_reg_labels)
    all_reg_preds = np.nan_to_num(all_reg_preds)
    all_reg_labels = all_reg_labels.flatten()
    all_reg_preds = all_reg_preds.flatten()

    reg_pearson, _ = pearsonr(all_reg_labels, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_labels, all_reg_preds)  # ����

    all_reg_preds_discrete = np.round(all_reg_preds).astype(int)
    all_reg_labels_discrete = np.round(all_reg_labels).astype(int)

    # ����ָ��
    cls_accuracy = accuracy_score(all_cls_labels, all_cls_preds)
    cls_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
    cls_recall = recall_score(all_cls_labels, all_cls_preds, average='macro')

    return (avg_total_loss,
            reg_pearson, reg_rmse, reg_mae, cls_accuracy, cls_f1, cls_recall)


#
def cross_validation(
        model_save_path,
        train_data_path,  # ѵ����·��
        val_data_path,  # ����֤��·��
        test_data_path,  # ���Լ�·��
        num_epochs=50,
        batch_size=25,
        learning_rate=2e-5,
        random_state=20
):
    # ===== ����Ԥ���ֵ����ݼ� =====
    # ����ѵ����
    train_dataset = dataloader.QADataset(
        r"train.txt",  # ʹ��Ԥ�����ѵ����·��

        tokenizer_name="bert-base",
        max_length=128
    )
    # ������֤��
    val_dataset = dataloader.QADataset(
        r"val.txt",  # ʹ��Ԥ�������֤��·��

        tokenizer_name="bert-base",
        max_length=128
    )
    # ���ز��Լ�
    test_dataset = dataloader.QADataset(
        r"test.txt",  # ʹ��Ԥ����Ĳ��Լ�·��

        tokenizer_name="bert-base",
        max_length=128
    )

    # ������ݼ��Ƿ�Ϊ��
    for name, dataset in [("ѵ����", train_dataset), ("��֤��", val_dataset), ("���Լ�", test_dataset)]:
        if len(dataset) == 0:
            raise ValueError(f"{name}Ϊ�գ���������·����")
    print(f"ѵ��������: {len(train_dataset)} | ��֤������: {len(val_dataset)} | ����������: {len(test_dataset)}")

    # ===== ����DataLoader =====
    # ע�⣺��֤���Ͳ��Լ�����Ҫshuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # ��ѵ������Ҫshuffle
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # ��֤���ر�shuffle
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # ���Լ��ر�shuffle
        drop_last=True
    )

    # ===== �豸���� =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== ģ�ͳ�ʼ�� =====
    subnet = md.Subnet()
    model = md.SNN(subnet)
    model.to(device)  # ʹ�ö�̬�豸����
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # ���ڳ���
        eta_min=1e-6  # ��Сѧϰ��
    )

    # ===== �����¼ =====
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

    # ===== ѵ��ѭ�� =====
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        # ѵ���׶�
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        train_loss, train_pearson, train_rmse, train_mae, train_acc, train_f1, train_recall = train_metrics

        # ��֤�׶�
        val_metrics = evaluate_model(model, val_loader, device, epoch, num_epochs)
        val_loss, val_pearson, val_rmse, val_mae, val_acc, val_f1, val_recall = val_metrics

        # ÿ��epoch��������ã�����Ҫָ�꣩��
        scheduler.step()
        # ��¼��ǰѧϰ��
        current_lr = optimizer.param_groups[0]['lr']

        # ѵ��ָ��
        train_results['total_loss'].append(train_loss)
        train_results['reg_pearson'].append(train_pearson)
        train_results['reg_rmse'].append(train_rmse)
        train_results['reg_mae'].append(train_mae)
        train_results['lr'].append(current_lr)
        train_results['cls_accuracy'].append(train_acc)
        train_results['cls_f1'].append(train_f1)
        train_results['cls_recall'].append(train_recall)

        # ��ָ֤��
        val_results['total_loss'].append(val_loss)
        val_results['reg_pearson'].append(val_pearson)
        val_results['reg_rmse'].append(val_rmse)
        val_results['reg_mae'].append(val_mae)
        val_results['lr'].append(current_lr)
        val_results['cls_accuracy'].append(val_acc)
        val_results['cls_f1'].append(val_f1)
        val_results['cls_recall'].append(val_recall)

        # �������ģ��
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(model_save_path, "best_model.pt"))
            print(f"���ģ���ѱ��� (��֤��rmse: {val_rmse:.4f})")

        # ��ӡ����
        print(f"[ѵ����] ��ʧ: {train_loss:.4f} | acc: {train_rmse:.4f}")
        print(f"[��֤��] ��ʧ: {val_loss:.4f} | acc: {val_rmse:.4f}")

    # ===== ����ѵ����� =====
    results_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        **{f'train_{k}': v for k, v in train_results.items()},
        **{f'val_{k}': v for k, v in val_results.items()}
    })
    results_df.to_csv(os.path.join(model_save_path, "training_results4_reg_cls_rmse_model1.csv"), index=False)

    # ===== ���ղ��� =====
    print("\n===== �ڲ��Լ����������� =====")
    model.load_state_dict(
        torch.load(os.path.join(model_save_path, "best_model.pt"), map_location=device, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device, epoch, num_epochs)
    test_loss, test_pearson, test_rmse, test_mae, test_acc, test_f1, test_recall = test_metrics

    # ��ӡ���Խ��
    print(f"[���Լ�] ����ʧ: {test_loss:.4f}")
    print(
        f"        Pearsonϵ��: {test_pearson:.4f} | rMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} ")  # �������
    print(f"        precision��ȷ��: {test_acc:.4f} | F1����: {test_f1:.4f} | �ٻ���: {test_recall:.4f}")

    # �������ղ��Խ��
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

    # ����Ԥ�������ݼ�·�����£�����ʵ������޸�
    cross_validation(
        model_save_path=model_save_directory,
        train_data_path=r"train.txt",  # Ԥ����ѵ����
        val_data_path=r"val.txt",  # Ԥ������֤��
        test_data_path=r"test.txt",  # Ԥ���ֲ��Լ�
        num_epochs=50,
        batch_size=25,
        learning_rate=2e-5,
        random_state=42
    )

