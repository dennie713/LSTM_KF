class LSTMConfig:
    def __init__(self):
        self.x_input_size = 7
        self.x_output_size = 3

        self.P_input_size = 18
        self.P_output_size = 9

        self.hidden_size = 64
        self.num_layers = 4
        self.dropout = 0.0

    def getLSTMConfig(self):
        return self.x_input_size, self.x_output_size, self.hidden_size, self.num_layers, self.dropout, self.P_input_size, self.P_output_size
