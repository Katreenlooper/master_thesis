import torch
import torch.nn.functional as F
import torch.nn as nn

#https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#define-models
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, hidden=None):
        outputs, hidden = self.gru(input_seq, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class LuongAttnEncoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, n_layers=1, dropout=0):
        super(LuongAttnEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, encoder_outputs, hidden=None):
        outputs, hidden = self.gru(input_seq, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        _outputs = outputs.clone()
        attn_weights = self.attn(outputs, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        _outputs = torch.mean(_outputs, 0)
        concat_input = torch.cat((_outputs, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        _outputs = self.out(concat_output)
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs1, encoder_outputs2):
        rnn_output, hidden = self.gru(input_step, last_hidden)
        attn_weights1 = self.attn(rnn_output, encoder_outputs1)
        context1 = attn_weights1.bmm(encoder_outputs1.transpose(0, 1))
        context1 = context1.squeeze(1)
        attn_weights2 = self.attn(rnn_output, encoder_outputs2)
        context2 = attn_weights2.bmm(encoder_outputs2.transpose(0, 1))
        context2 = context2.squeeze(1)
        rnn_output = rnn_output.squeeze(0)
        concat_input = torch.cat((rnn_output, context1, context2), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)

        return output, hidden