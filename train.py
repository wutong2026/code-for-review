import torch
import torch.nn as nn
import fire
import pandas as pd
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models import get_model_class
from sklearn.preprocessing import LabelEncoder
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
import csv
from collections import Counter


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(output, tgt)
            total_loss += loss.item()

            # 计算准确率
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)

    return total_loss / len(test_dataloader), correct / total


def load_code_dataset(data_path, batch_size, max_features=1000, sample_ratio=0.1):
    """加载代码分类数据集 - 简化版"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {os.path.abspath(data_path)}")

    try:
        # 为了节省内存，只读取一部分数据
        print("正在读取数据...")
        df = pd.read_csv(data_path, nrows=int(50000 * 5) if sample_ratio < 1 else None)

        # 如果数据太大，进行采样
        if len(df) > 100000 and sample_ratio < 1:
            df = df.sample(frac=sample_ratio, random_state=42)

        print(f"✓ 成功读取CSV文件，总行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"读取CSV文件失败: {str(e)}")

    # 检查必要的列是否存在
    required_columns = ['project', 'func', 'target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")

    # 查看数据基本情况
    print("\n" + "=" * 60)
    print("数据统计信息:")
    print(f"项目类别数量: {df['project'].nunique()}")
    print(f"目标类别分布:")
    target_counts = df['target'].value_counts()
    print(target_counts)
    print(f"函数平均长度: {df['func'].astype(str).str.len().mean():.0f} 字符")

    # 1. 对A列（project）进行简单的数字编码而不是one-hot
    print("\n对project列进行编码...")
    project_encoder = LabelEncoder()
    project_encoded = project_encoder.fit_transform(df['project'].values)

    # 将编码转换为0-1范围内的值
    project_normalized = project_encoded.astype(np.float32) / max(1, project_encoded.max())

    print(f"项目类别数量: {len(project_encoder.classes_)}")

    # 2. 对B列（func）进行简单的特征提取
    print("\n提取代码特征...")

    def extract_simple_features(text):
        """提取简单的代码特征"""
        if not isinstance(text, str):
            text = str(text)

        features = []

        # 1. 代码长度
        features.append(len(text) / 1000.0)  # 归一化

        # 2. 行数
        features.append(text.count('\n') / 10.0)

        # 3. 括号数量
        features.append((text.count('(') + text.count(')')) / 10.0)

        # 4. 分号数量
        features.append(text.count(';') / 5.0)

        # 5. 大括号数量
        features.append((text.count('{') + text.count('}')) / 5.0)

        # 6. 关键字数量（简单的统计）
        keywords = ['if', 'for', 'while', 'return', 'static', 'struct', 'int', 'void']
        keyword_count = sum(text.lower().count(keyword) for keyword in keywords)
        features.append(keyword_count / 5.0)

        return np.array(features, dtype=np.float32)

    # 并行提取特征
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("并行提取特征中...")
    func_features = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_simple_features, text): i
                   for i, text in enumerate(df['func'].values)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="提取特征"):
            features = future.result()
            func_features.append(features)

    func_features = np.vstack(func_features)
    print(f"文本特征维度: {func_features.shape}")

    # 3. 合并特征
    X = np.hstack([
        project_normalized.reshape(-1, 1),  # project编码
        func_features  # 代码特征
    ])
    y = df['target'].values.astype(np.float32).reshape(-1, 1)  # 二分类目标

    print(f"总特征维度: {X.shape[1]}")
    print(f"类别分布: 0: {np.sum(y == 0)}, 1: {np.sum(y == 1)}")

    # 4. 划分数据集
    total_samples = len(X)
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * 0.8)
    val_size = int(total_samples * 0.1)
    test_size = total_samples - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"\n划分后类别分布:")
    print(f"训练集: 0: {np.sum(y_train == 0)}, 1: {np.sum(y_train == 1)}")
    print(f"验证集: 0: {np.sum(y_val == 0)}, 1: {np.sum(y_val == 1)}")
    print(f"测试集: 0: {np.sum(y_test == 0)}, 1: {np.sum(y_test == 1)}")

    # 5. 对训练集进行简单的过采样
    print("\n对训练集进行简单过采样...")

    # 计算类别不平衡比例
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    if len(class_1_indices) > 0:
        # 对少数类（class 1）进行过采样
        oversample_factor = len(class_0_indices) // len(class_1_indices)
        if oversample_factor > 1:
            oversampled_indices = []
            for _ in range(oversample_factor):
                oversampled_indices.extend(class_1_indices)

            # 如果还需要更多样本，随机选择一些
            remaining = len(class_0_indices) - len(oversampled_indices)
            if remaining > 0:
                extra_indices = np.random.choice(class_1_indices, remaining, replace=True)
                oversampled_indices.extend(extra_indices)

            # 合并过采样后的索引
            train_indices_resampled = np.concatenate([class_0_indices, oversampled_indices])
            X_train_resampled = X_train[train_indices_resampled]
            y_train_resampled = y_train[train_indices_resampled]

            print(f"过采样后训练集: 0: {np.sum(y_train_resampled == 0)}, 1: {np.sum(y_train_resampled == 1)}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
            print("类别相对平衡，不进行过采样")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("少数类样本为0，不进行过采样")

    # 6. 数据归一化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 创建数据集类
    class CodeDataset(Dataset):
        def __init__(self, features, labels, training=False):
            self.features = features.astype(np.float32)
            self.labels = labels.astype(np.float32)
            self.training = training

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            features = torch.tensor(self.features[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.training and random.random() < 0.3:  # 30%的概率添加噪声
                noise = torch.randn_like(features) * 0.01
                features += noise

            return features, label

    # 创建数据集
    train_dataset = CodeDataset(X_train_scaled, y_train_resampled, training=True)
    val_dataset = CodeDataset(X_val_scaled, y_val, training=False)
    test_dataset = CodeDataset(X_test_scaled, y_test, training=False)

    # 创建DataLoader
    def create_dataloader(dataset, batch_size, shuffle):
        valid_batch_size = min(batch_size, len(dataset))
        if valid_batch_size != batch_size:
            print(f"警告: 自动调整batch_size {batch_size} → {valid_batch_size}")
        return DataLoader(
            dataset,
            batch_size=valid_batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0  # 设置为0避免多进程问题
        )

    train_dataloader = create_dataloader(train_dataset, batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("数据加载完成！")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"输入特征维度: {X.shape[1]}")
    print("=" * 60)

    return (train_dataset, val_dataset, test_dataset,
            train_dataloader, val_dataloader, test_dataloader,
            project_encoder, scaler)


def save_predictions_csv(predictions, targets, filepath):
    """保存预测结果到CSV文件"""
    df_results = pd.DataFrame({
        'true_label': targets.flatten(),
        'predicted_label': predictions.flatten(),
        'predicted_prob': predictions.flatten()  # 保存概率值
    })
    df_results.to_csv(filepath, index=False)
    print(f"✓ 预测结果已保存到: {filepath}")


def save_metrics_to_csv(metrics, filepath):
    """保存评估指标到CSV文件"""
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(filepath, index=False)
    print(f"✓ 评估指标已保存到: {filepath}")


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir='output'):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(val_accs, label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存训练历史到CSV
    df_history = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    df_history.to_csv(f'{save_dir}/training_history.csv', index=False)


def evaluate_classification(model, test_dataloader, device, save_dir='output'):
    """评估分类模型并生成各种图表"""
    model.eval()
    all_predictions = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 确保标签是整数类型
    all_predictions_int = all_predictions.astype(int)
    all_targets_int = all_targets.astype(int)

    # 计算评估指标
    accuracy = accuracy_score(all_targets_int, all_predictions_int)
    conf_matrix = confusion_matrix(all_targets_int, all_predictions_int)

    # 获取类别标签
    unique_labels = sorted(np.unique(np.concatenate([all_targets_int, all_predictions_int])))

    # 生成分类报告
    report = classification_report(
        all_targets_int,
        all_predictions_int,
        target_names=[str(label) for label in unique_labels],
        output_dict=True
    )

    # 保存预测结果
    save_predictions_csv(all_predictions, all_targets, f'{save_dir}/test_predictions.csv')

    # 计算ROC曲线（需要确保有至少两个类别）
    if len(np.unique(all_targets_int)) >= 2:
        fpr, tpr, thresholds = roc_curve(all_targets_int, all_probs)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, thresholds = [], [], []
        roc_auc = 0.5
        print("警告: 测试集中只有一个类别，无法计算ROC曲线")

    # 1. 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.colorbar()

    classes = [str(label) for label in unique_labels]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在矩阵中显示数值
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 绘制ROC曲线
    if len(fpr) > 0 and len(tpr) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 准备详细的评估指标
    detailed_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
    }

    # 添加每个类别的指标
    for label in unique_labels:
        label_str = str(label)
        if label_str in report and isinstance(report[label_str], dict):
            detailed_metrics[f'precision_class_{label_str}'] = report[label_str].get('precision', 0.0)
            detailed_metrics[f'recall_class_{label_str}'] = report[label_str].get('recall', 0.0)
            detailed_metrics[f'f1_class_{label_str}'] = report[label_str].get('f1-score', 0.0)

    # 添加平均指标
    if 'macro avg' in report:
        detailed_metrics['macro_avg_precision'] = report['macro avg']['precision']
        detailed_metrics['macro_avg_recall'] = report['macro avg']['recall']
        detailed_metrics['macro_avg_f1'] = report['macro avg']['f1-score']

    if 'weighted avg' in report:
        detailed_metrics['weighted_avg_precision'] = report['weighted avg']['precision']
        detailed_metrics['weighted_avg_recall'] = report['weighted avg']['recall']
        detailed_metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']

    # 添加整体统计
    detailed_metrics['num_samples'] = len(all_targets)
    detailed_metrics['num_classes'] = len(unique_labels)
    detailed_metrics['class_distribution'] = str(dict(zip(unique_labels, np.bincount(all_targets_int.flatten()))))

    # 保存指标
    save_metrics_to_csv(detailed_metrics, f'{save_dir}/test_metrics.csv')

    # 打印评估结果
    print("\n" + "=" * 60)
    print("测试集评估结果:")
    print(f"准确率: {accuracy:.4f}")
    if roc_auc != 0.5:
        print(f"AUC分数: {roc_auc:.4f}")
    print(f"类别分布: {detailed_metrics['class_distribution']}")
    print(f"混淆矩阵:\n{conf_matrix}")
    print(f"分类报告:\n{classification_report(all_targets_int, all_predictions_int)}")
    print("=" * 60)

    return detailed_metrics


def main(
        dataset="data/dataset.csv",
        model_type="mambaplus",
        epochs=50,
        batch_size=16,
        lr=1e-5,
        train=True,
        test=True,
        sample_ratio=0.1
):
    os.makedirs("output", exist_ok=True)
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 记录实验配置
    config = {
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'device': str(device),
        'sample_ratio': sample_ratio
    }

    # 加载数据集
    (train_dataset, val_dataset, test_dataset,
     train_dataloader, val_dataloader, test_dataloader,
     project_encoder, scaler) = load_code_dataset(dataset, batch_size, sample_ratio=sample_ratio)

    # 获取输入维度
    sample_features, _ = train_dataset[0]
    input_size = sample_features.shape[0]
    output_size = 1  # 二分类任务

    print(f"模型配置: 输入维度={input_size}, 输出维度={output_size}")

    # 模型初始化
    model_cls = get_model_class(model_type)
    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        hidden_size=128,  # 隐藏层大小
        num_layers=4,  # 层数
        dropout=0.3
    ).to(device)

    print(f"创建 {model_type} 模型，参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 计算类别权重
    class_counts = Counter(train_dataset.labels.flatten())
    pos_weight = torch.tensor([class_counts[0] / max(1, class_counts[1])]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01
    )

    # 训练循环
    if train:
        best_val_loss = float('inf')
        best_val_acc = 0.0
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        # 初始化训练日志
        log_file = open('output/training_log.csv', 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            total_trloss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for features, labels in progress_bar:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                # 计算准确率
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct = (preds == labels).sum().item()

                total_trloss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / labels.size(0):.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

            train_loss = total_trloss / len(train_dataloader)
            train_acc = total_correct / total_samples

            # 验证阶段
            val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
            current_lr = optimizer.param_groups[0]['lr']

            # 更新学习率
            scheduler.step()

            # 记录历史
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # 写入日志
            log_writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc])
            log_file.flush()

            # 每2个epoch绘制一次训练历史
            if (epoch + 1) % 2 == 0:
                plot_training_history(train_loss_history, val_loss_history,
                                      train_acc_history, val_acc_history)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config,
                    'project_encoder': project_encoder,
                    'scaler': scaler
                }, 'output/best_model.pth')
                print(f"✓ 保存最佳模型: Epoch {epoch + 1}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")

            print(f'Epoch {epoch + 1}/{epochs} | '
                  f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
                  f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | '
                  f'LR: {current_lr:.2e}')

        # 保存最终模型
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': config
        }, 'output/final_model.pth')

        # 绘制最终的训练历史
        plot_training_history(train_loss_history, val_loss_history,
                              train_acc_history, val_acc_history)

        log_file.close()

    # 测试阶段
    if test:
        # 加载最佳模型
        if os.path.exists('output/best_model.pth'):
            checkpoint = torch.load('output/best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 加载最佳模型 (Epoch {checkpoint['epoch'] + 1}, "
                  f"Val Loss: {checkpoint['val_loss']:.6f}, "
                  f"Val Acc: {checkpoint['val_acc']:.4f})")
        else:
            print("警告：未找到最佳模型，使用最终模型进行测试")
            checkpoint = torch.load('output/final_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])

        # 执行评估
        evaluate_classification(model, test_dataloader, device)

        print("✓ 测试完成！所有结果已保存到output目录")


if __name__ == "__main__":
    fire.Fire(main)