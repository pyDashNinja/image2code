import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # The input image is 16x16, so we will use a 4x4 grid of attention blocks
        self.num_blocks = 4
        self.num_heads = 8

        # The input and output of each attention block is a feature map of size 256
        self.dim_model = 256
        self.dim_keys = self.dim_model // self.num_heads
        self.dim_values = self.dim_model // self.num_heads

        # The feedforward network consists of two fully-connected layers with ReLU activation
        self.feedforward = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model * 4),
            nn.ReLU(),
            nn.Linear(self.dim_model * 4, self.dim_model)
        )

        # The attention blocks are implemented as a 2D grid of self-attention layers
        self.attention_blocks = nn.ModuleList([
            SelfAttention(self.dim_model, self.num_heads)
            for _ in range(self.num_blocks ** 2)
        ])

        # The final fully-connected layer maps the output of the attention blocks to the desired number of classes
        self.output_layer = nn.Linear(self.dim_model, num_classes)

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Flatten the input image into a sequence of tokens
        x = x.view(batch_size, -1, height * width)

        # The input to the transformer is a sequence of tokens with one feature each
        x = x.transpose(1, 2)
        
        # Apply the attention blocks and feedforward network in a loop
        for _ in range(self.num_blocks):
            x = self.attention_blocks(x)
            x = self.feedforward(x)

        # Reshape the output of the transformer to the original image size
        x = x.view(batch_size, -1, height, width)

        # Apply the final fully-connected layer to each pixel of the image
        x = self.output_layer(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_keys = dim_model // num_heads
        self.dim_values = dim_model // num_heads

        # The key, query, and value projections are fully-connected layers
        self.key_projection = nn.Linear(dim_model, dim_model)
        self.query_projection = nn.Linear(dim_model, dim_model)
        self.value_projection = nn.Linear(dim_model, dim_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        # Project the input sequence onto the key, query, and value spaces
        keys = self.key_projection(x)
        queries = self.query_projection(x)
        values = self.value_projection(x)

        # Split the key, query, and value tensors into num_heads slices along the batch dimension
        keys = keys.view(batch_size, seq_length, self.num_heads, self.dim_keys)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.dim_keys)
        values = values.view(batch_size, seq_length, self.num_heads, self.dim_values)

        # Transpose the slices so that the attention dim is the last dimension
        keys = keys.transpose(2, 3)
        queries = queries.transpose(2, 3)
        values = values.transpose(2, 3)

        # Compute the dot product of the queries with the keys, and normalize the scores
        scores = torch.matmul(queries, keys) / (self.dim_keys ** 0.5)
        scores = F.softmax(scores, dim=-1)

        # Multiply the values by the normalized scores and sum the result
        weighted_values = torch.matmul(scores, values)
        weighted_values = weighted_values.transpose(2, 3)
        weighted_values = weighted_values.view(batch_size, seq_length, self.dim_model)

        # Add the weighted values to the input to obtain the attention output
        output = weighted_values + x

        return output

