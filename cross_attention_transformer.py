import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        embed_dim,
        num_heads=8,
        dropout=0.1,
        bias=True,
    ):
        """
        Cross Attention Module

        Args:
            query_dim (int): Dimension of query input
            key_dim (int): Dimension of key input
            value_dim (int): Dimension of value input
            embed_dim (int): Embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            bias (bool): Whether to use bias in linear layers
        """
        super(CrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(query_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(key_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(value_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, value_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Forward pass of cross attention

        Args:
            query (Tensor): Query tensor of shape (batch_size, query_len, query_dim)
            key (Tensor): Key tensor of shape (batch_size, key_len, key_dim)
            value (Tensor): Value tensor of shape (batch_size, value_len, value_dim)
            attn_mask (Tensor, optional): Attention mask of shape (query_len, key_len) or
                                        (batch_size, query_len, key_len)
            key_padding_mask (Tensor, optional): Key padding mask of shape (batch_size, key_len)

        Returns:
            output (Tensor): Output tensor of shape (batch_size, query_len, embed_dim)
            attn_weights (Tensor): Attention weights of shape (batch_size, num_heads, query_len, key_len)
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        value_len = value.size(1)

        # Linear projections
        Q = self.query_proj(query)  # (batch_size, query_len, embed_dim)
        K = self.key_proj(key)  # (batch_size, key_len, embed_dim)
        V = self.value_proj(value)  # (batch_size, value_len, embed_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, value_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, query_len, key_len)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Broadcast mask for all batches and heads
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # Broadcast mask for all heads
                attn_mask = attn_mask.unsqueeze(1)

            attn_scores.masked_fill_(attn_mask == 0, float("-inf"))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask shape: (batch_size, key_len)
            # Reshape to (batch_size, 1, 1, key_len) for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores.masked_fill_(key_padding_mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(
            attn_weights
        )  # (batch_size, num_heads, query_len, key_len)
        attn_weights = attn_weights.squeeze(2)  # remove query dimension
        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, num_heads, key_len, 1)
        attn_weights = attn_weights.expand(
            -1, -1, -1, self.head_dim
        )  # (batch_size, num_heads, key_len, head_dim)

        # Apply attention to values
        # attn_output = torch.matmul(attn_weights, V)
        # print(attn_weights.transpose(-2, -1).shape)
        # print(V.shape)
        # attn_output = torch.matmul(attn_weights.transpose(-2, -1), V)
        attn_output = V * attn_weights
        # print(attn_output.shape)
        # Shape: (batch_size, num_heads, key_len, head_dim)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous()
            # .view(batch_size, query_len, self.embed_dim)
            .view(batch_size, value_len, self.embed_dim)
        )

        # Final output projection
        output = self.out_proj(attn_output)
        # output = attn_output

        return output, attn_weights


class CrossAttentionEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with cross attention capability
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        has_cross_attention=False,
        cross_attention_dim=None,
    ):
        super(CrossAttentionEncoderLayer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.has_cross_attention = has_cross_attention

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Cross-attention (optional)
        if has_cross_attention:
            cross_dim = (
                cross_attention_dim if cross_attention_dim is not None else d_model
            )
            self.cross_attn = CrossAttention(
                query_dim=d_model,
                key_dim=cross_dim,
                value_dim=cross_dim,
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
            )
            self.norm_cross = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout_cross = nn.Dropout(dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        """Get activation function"""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swish":
            return F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        cross_src=None,
        cross_src_mask=None,
        cross_src_key_padding_mask=None,
    ):
        """
        Args:
            src: Source sequence (batch_size, seq_len, d_model)
            src_mask: Self-attention mask
            src_key_padding_mask: Padding mask for source
            cross_src: Cross-attention source (batch_size, cross_seq_len, cross_dim)
            cross_src_mask: Cross-attention mask
            cross_src_key_padding_mask: Padding mask for cross source
        """
        # Self-attention
        src2, self_attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Cross-attention (if enabled and cross_src is provided)
        cross_attn_weights = None
        if self.has_cross_attention and cross_src is not None:
            src2, cross_attn_weights = self.cross_attn(
                src,
                cross_src,
                cross_src,
                attn_mask=cross_src_mask,
                key_padding_mask=cross_src_key_padding_mask,
            )
            src = src + self.dropout_cross(src2)
            src = self.norm_cross(src)

        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, self_attn_weights, cross_attn_weights


class CrossAttentionTransformerEncoder(nn.Module):
    """
    Complete transformer encoder with cross attention capability
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        has_cross_attention=False,
        cross_attention_dim=None,
        return_attention_weights=False,
    ):
        super(CrossAttentionTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.has_cross_attention = has_cross_attention
        self.return_attention_weights = return_attention_weights

        # Create encoder layers
        self.layers = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    has_cross_attention=has_cross_attention,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Positional encoding (optional)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        cross_src=None,
        cross_mask=None,
        cross_key_padding_mask=None,
        use_pos_encoding=True,
    ):
        """
        Args:
            src: Source sequence (batch_size, seq_len, d_model)
            mask: Self-attention mask
            src_key_padding_mask: Padding mask for source
            cross_src: Cross-attention source (batch_size, cross_seq_len, cross_dim)
            cross_mask: Cross-attention mask
            cross_key_padding_mask: Padding mask for cross source
            use_pos_encoding: Whether to apply positional encoding

        Returns:
            output: Encoded output (batch_size, seq_len, d_model)
            attention_weights: Dict of attention weights (if return_attention_weights=True)
        """
        # Apply positional encoding
        if use_pos_encoding:
            src = self.pos_encoding(src)

        output = src
        all_self_attn_weights = []
        all_cross_attn_weights = []

        # Pass through encoder layers
        for layer in self.layers:
            output, self_attn_weights, cross_attn_weights = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                cross_src=cross_src,
                cross_src_mask=cross_mask,
                cross_src_key_padding_mask=cross_key_padding_mask,
            )

            if self.return_attention_weights:
                all_self_attn_weights.append(self_attn_weights)
                if cross_attn_weights is not None:
                    all_cross_attn_weights.append(cross_attn_weights)

        # Final layer normalization
        output = self.norm(output)

        if self.return_attention_weights:
            attention_weights = {
                "self_attention": all_self_attn_weights,
                "cross_attention": (
                    all_cross_attn_weights if all_cross_attn_weights else None
                ),
            }
            return output, attention_weights

        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        pe = self.pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


if __name__ == "__main__":
    # Example usage
    model = CrossAttention(
        query_dim=768,
        key_dim=768 + 64,
        value_dim=768 + 64,
        embed_dim=768 + 64,
        num_heads=8,
        dropout=0.1,
        bias=True,
    )

    query = torch.randn(2, 1, 768)
    key = torch.randn(2, 10, 768 + 64)
    value = torch.randn(2, 10, 768 + 64)

    output, _ = model(query=query, key=key, value=value)
    print(output.shape)
