import torch
import torchnlp

class TransformerDecoder(torchnlp.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.self_attention = torchnlp.nn.MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.encoder_attention = torchnlp.nn.MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = torchnlp.nn.Sequential(
            torchnlp.nn.Linear(d_model, d_ff),
            torchnlp.nn.ReLU(),
            torchnlp.nn.Linear(d_ff, d_model)
        )
        self.norm1 = torchnlp.nn.LayerNorm(d_model)
        self.norm2 = torchnlp.nn.LayerNorm(d_model)
        self.dropout1 = torchnlp.nn.Dropout(dropout)
        self.dropout2 = torchnlp.nn.Dropout(dropout)

    def forward(self, inputs, encoder_outputs, mask=None):
        # self attention
        x = self.self_attention(inputs, inputs, inputs, mask=mask)[0]
        x = self.dropout1(x)
        x = self.norm1(x + inputs)
        # encoder attention
        x = self.encoder_attention(x, encoder_outputs, encoder_outputs)[0]
        x = self.dropout2(x)
        x = self.norm2(x + inputs)
        # feed forward
        x = self.feed_forward(x)
        return x
