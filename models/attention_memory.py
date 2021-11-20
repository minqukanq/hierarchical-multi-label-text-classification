import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, num_units, attention_unit_size, num_classes):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Linear(num_units, attention_unit_size, bias=False)
        self.fc2 = nn.Linear(attention_unit_size, num_classes, bias=False)

    def forward(self, input_x):
        attention_matrix = self.fc2(torch.tanh(self.fc1(input_x))).transpose(1, 2)
        attention_weight = torch.softmax(attention_matrix, dim=-1)
        attention_out = torch.matmul(attention_weight, input_x)
        return attention_weight, torch.mean(attention_out, dim=1)


class LocalLayer(nn.Module):
    def __init__(self, num_units, num_classes):
        super(LocalLayer, self).__init__()
        self.fc = nn.Linear(num_units, num_classes)

    def forward(self, input_x, input_att_weight):
        logits = self.fc(input_x)
        scores = torch.sigmoid(logits)
        visual = torch.mul(input_att_weight, scores.unsqueeze(-1))
        visual = torch.softmax(visual, dim=-1)
        visual = torch.mean(visual, dim=1)
        return logits, scores, visual