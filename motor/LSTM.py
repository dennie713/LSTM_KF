import torch
import torch.nn as nn

class LSTM_KF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTM_KF, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size , output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 只取最後一個時間步的輸出
        return output