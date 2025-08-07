
import torch
import torch.nn as nn

# 自动选择设备：GPU优先，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CRF(nn.Module):
    def __init__(self, num_tags, pad_id, start_id, end_id):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id

        # 初始化转移矩阵并进行约束
        self.transition = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.transition.data[:, self.start_id] = -10000.0
        self.transition.data[self.end_id, :] = -10000.0
        self.transition.data[:, self.pad_id] = -10000.0
        self.transition.data[self.pad_id, :] = -10000.0
        self.transition.data[self.pad_id, self.pad_id] = 0.0
        self.transition.data[self.pad_id, :self.end_id] = 0.0

    def forward(self, emissions, tags, mask):
        gold_score = self.compute_gold_score(emissions, tags, mask)
        total_score = self.compute_total_score(emissions, mask)
        return torch.sum(total_score - gold_score)

    def compute_gold_score(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        score = torch.zeros(batch_size, device=device)
        start_tag = tags[:, 0]
        last_tag_index = mask.sum(1) - 1
        last_tag_index = last_tag_index.long()
        last_tag = torch.gather(tags, 1, last_tag_index.unsqueeze(1)).squeeze(1)

        # 初始转移 + 第一个标签分数
        score += self.transition[self.start_id, start_tag] + emissions[:, 0, :].gather(1, start_tag.unsqueeze(
            1)).squeeze(1)

        for i in range(1, seq_len):
            current_tag = tags[:, i]
            prev_tag = tags[:, i - 1]
            trans_score = self.transition[prev_tag, current_tag]
            emit_score = emissions[:, i, :].gather(1, current_tag.unsqueeze(1)).squeeze(1)
            score += (trans_score + emit_score) * mask[:, i]

        score += self.transition[last_tag, self.end_id]
        return score

    def compute_total_score(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.transition[self.start_id, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = []
            for next_tag in range(num_tags):
                trans_score = self.transition[:, next_tag].unsqueeze(0)
                emit_score = emissions[:, i, next_tag].unsqueeze(1)
                score_t = trans_score + emit_score + score
                broadcast_score.append(torch.logsumexp(score_t, dim=1))

            new_score = torch.stack(broadcast_score).t()
            mask_i = mask[:, i].float().unsqueeze(-1)
            score = mask_i * new_score + (1 - mask_i) * score

        score += self.transition[:, self.end_id].unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def predict(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.transition[self.start_id, :].unsqueeze(0) + emissions[:, 0]

        backpointers = torch.arange(num_tags, device=device).view(1, -1, 1).repeat(batch_size, 1, 1)

        for i in range(1, seq_len):
            new_score = torch.zeros(batch_size, num_tags, device=device)
            new_pointer = torch.zeros(batch_size, num_tags, 1, device=device)
            for next_tag in range(num_tags):
                trans_score = self.transition[:, next_tag].unsqueeze(0)
                emit_score = emissions[:, i, next_tag].unsqueeze(1)
                score_t = trans_score + emit_score + score
                max_score, pointer = torch.max(score_t, dim=1)
                new_score[:, next_tag] = max_score
                new_pointer[:, next_tag, 0] = pointer

            mask_i = mask[:, i].float().unsqueeze(-1)
            score = mask_i * new_score + (1 - mask_i) * score

            tmp_index = torch.arange(num_tags, device=device).view(1, -1, 1).repeat(batch_size, 1, 1)
            pad_pointer = torch.ones(batch_size, num_tags, 1, device=device) * self.pad_id
            append_pointer = mask_i.unsqueeze(-1) * tmp_index + (1 - mask_i.unsqueeze(-1)) * pad_pointer

            new_pointer = new_pointer.long()
            prev_tags = backpointers[torch.arange(batch_size).unsqueeze(1).repeat(1, num_tags),
                        new_pointer[:, :, 0], :]
            backpointers = torch.cat([prev_tags, append_pointer], dim=-1)

        final_score = score + self.transition[:, self.end_id].unsqueeze(0)
        _, best_last_tag = torch.max(final_score, dim=1)
        return backpointers[torch.arange(batch_size), best_last_tag]

class NER(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, num_tags, pad_id, start_id, end_id, weight=None,
                 dropout=0.3):
        super(NER, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 词向量层
        if weight is None:
            embedding_weights = nn.init.xavier_normal_(torch.empty(vocab_size, embedding_dim))
            self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=embedding_weights)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=weight)

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, pad_id, start_id, end_id)

    def forward(self, inputs, tags, mask):
        inputs, tags, mask = inputs.to(device), tags.to(device), mask.to(device)
        embeds = self.embedding(inputs)
        embeds = self.dropout(embeds)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)
        loss = self.crf(emissions, tags, mask)
        return loss

    def predict(self, inputs, mask):
        inputs, mask = inputs.to(device), mask.to(device)
        embeds = self.embedding(inputs)
        embeds = self.dropout(embeds)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)
        return self.crf.predict(emissions, mask)
