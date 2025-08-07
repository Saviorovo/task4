import torch
from torch import optim
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Neural_network import NER
from prework import get_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def train_ner_model(model, train_data, test_data, lr, epochs, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_record = []
    test_loss_record = []
    train_accuracy_record = []
    test_accuracy_record = []
    train_precision_record = []
    test_precision_record = []
    train_recall_record = []
    test_recall_record = []
    train_f1_record = []
    test_f1_record = []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask = (y_batch != 0).float().to(device)  # 确保mask是float类型

            loss = model(x_batch, y_batch, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        # 初始化评估指标容器
        all_train_true, all_train_pred = [], []
        all_test_true, all_test_pred = [], []
        train_loss, test_loss = 0, 0

        # 验证训练集表现
        with torch.no_grad():
            for batch in train_data:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask = (y_batch != 0).float().to(device)

                loss = model(x_batch, y_batch, mask)
                train_loss += loss.item() / batch_size / y_batch.shape[1]
                pred = model.predict(x_batch, mask)

                # 收集有效标签
                valid_indices = mask.bool()
                all_train_true.extend(y_batch[valid_indices].cpu().numpy())
                all_train_pred.extend(pred[valid_indices].cpu().numpy())

        # 计算训练集指标
        train_accuracy = accuracy_score(all_train_true, all_train_pred) if len(all_train_true) > 0 else 0
        train_precision = precision_score(all_train_true, all_train_pred, average='macro', zero_division=0) if len(all_train_true) > 0 else 0
        train_recall = recall_score(all_train_true, all_train_pred, average='macro', zero_division=0) if len(all_train_true) > 0 else 0
        train_f1 = f1_score(all_train_true, all_train_pred, average='macro', zero_division=0) if len(all_train_true) > 0 else 0

        # 验证测试集表现
        with torch.no_grad():
            for batch in test_data:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask = (y_batch != 0).float().to(device)

                loss = model(x_batch, y_batch, mask)
                test_loss += loss.item() / batch_size / y_batch.shape[1]
                pred = model.predict(x_batch, mask)

                # 收集有效标签
                valid_indices = mask.bool()
                all_test_true.extend(y_batch[valid_indices].cpu().numpy())
                all_test_pred.extend(pred[valid_indices].cpu().numpy())

        # 计算测试集指标
        test_accuracy = accuracy_score(all_test_true, all_test_pred) if len(all_test_true) > 0 else 0
        test_precision = precision_score(all_test_true, all_test_pred, average='macro', zero_division=0) if len(all_test_true) > 0 else 0
        test_recall = recall_score(all_test_true, all_test_pred, average='macro', zero_division=0) if len(all_test_true) > 0 else 0
        test_f1 = f1_score(all_test_true, all_test_pred, average='macro', zero_division=0) if len(all_test_true) > 0 else 0

        # 记录结果
        train_loss_record.append(train_loss / len(train_data))
        test_loss_record.append(test_loss / len(test_data))
        train_accuracy_record.append(train_accuracy)
        test_accuracy_record.append(test_accuracy)
        train_precision_record.append(train_precision)
        test_precision_record.append(test_precision)
        train_recall_record.append(train_recall)
        test_recall_record.append(test_recall)
        train_f1_record.append(train_f1)
        test_f1_record.append(test_f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_data):.4f} | Test Loss: {test_loss/len(test_data):.4f}")
        print(f"Train Metrics - Acc: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test Metrics  - Acc: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    return (
        train_loss_record, test_loss_record,
        train_accuracy_record, test_accuracy_record,
        train_precision_record, test_precision_record,
        train_recall_record, test_recall_record,
        train_f1_record, test_f1_record
    )

def NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, lr, batch_size, epochs):
    # 获取训练数据和测试数据 batch
    train_data_random = get_batch(random_embedding.train_x_matrix, random_embedding.train_y_matrix, batch_size)
    test_data_random = get_batch(random_embedding.test_x_matrix, random_embedding.test_y_matrix, batch_size)
    train_data_glove = get_batch(glove_embedding.train_x_matrix, glove_embedding.train_y_matrix, batch_size)
    test_data_glove = get_batch(glove_embedding.test_x_matrix, glove_embedding.test_y_matrix, batch_size)

    # 初始化模型
    model_random = NER(len_feature, random_embedding.len_words, len_hidden,
                       random_embedding.len_tag, 0, 1, 2).to(device)
    model_glove = NER(len_feature, glove_embedding.len_words, len_hidden,
                      glove_embedding.len_tag, 0, 1, 2,
                      weight=torch.tensor(glove_embedding.embedding, dtype=torch.float)).to(device)

    # 训练模型并获取所有指标
    results_random = train_ner_model(model_random, train_data_random, test_data_random, lr, epochs, batch_size)
    results_glove = train_ner_model(model_glove, train_data_glove, test_data_glove, lr, epochs, batch_size)

    x = list(range(1, epochs + 1))
    plt.figure(figsize=(18, 12))

    # Loss 对比
    plt.subplot(3, 2, 1)
    plt.plot(x, results_random[0], 'r--', label='Random Train')
    plt.plot(x, results_glove[0], 'g--', label='GloVe Train')
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(x, results_random[1], 'r--', label='Random Test')
    plt.plot(x, results_glove[1], 'g--', label='GloVe Test')
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy 对比
    plt.subplot(3, 2, 3)
    plt.plot(x, results_random[2], 'r--', label='Random Train')
    plt.plot(x, results_glove[2], 'g--', label='GloVe Train')
    plt.title("Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(x, results_random[3], 'r--', label='Random Test')
    plt.plot(x, results_glove[3], 'g--', label='GloVe Test')
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    # 综合指标对比
    plt.subplot(3, 2, 5)
    plt.plot(x, results_random[4], 'r:', label='Precision (Train)')
    plt.plot(x, results_random[6], 'b:', label='Recall (Train)')
    plt.plot(x, results_random[8], 'g:', label='F1 (Train)')
    plt.title("Random Embedding Metrics (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(x, results_glove[4], 'r:', label='Precision (Test)')
    plt.plot(x, results_glove[6], 'b:', label='Recall (Test)')
    plt.plot(x, results_glove[8], 'g:', label='F1 (Test)')
    plt.title("GloVe Embedding Metrics (Test)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig('full_metrics_plot.jpg')
    plt.show()