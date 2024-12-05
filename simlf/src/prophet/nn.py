import torch
import torch.nn as nn
from torch import Tensor


def make_lstm(input_shape, hidden_size, batch_first=True, **kwargs):
    return (
        nn.LSTM(
            input_size=input_shape[-1], hidden_size=hidden_size, batch_first=batch_first, **kwargs
        ),
        input_shape[:-1] + [hidden_size],
    )


def make_linear(input_shape, out_features, **kwargs):
    return (
        nn.Linear(in_features=input_shape[-1], out_features=out_features, **kwargs),
        input_shape[:-1] + [out_features],
    )


def make_layer_norm(input_shape, axis=[-1], eps=1e-5, elementwise_affine=True, **kwargs):
    return (
        nn.LayerNorm(
            [input_shape[i] for i in axis], eps=eps, elementwise_affine=elementwise_affine, **kwargs
        ),
        input_shape,
    )


def make_dropout(input_shape, **kwargs):
    return (nn.Dropout(**kwargs), input_shape)


def make_tanh(input_shape, **kwargs):
    return (torch.tanh, input_shape)


FACTORY_MAP = {
    "LSTM": make_lstm,
    "Linear": make_linear,
    "LayerNorm": make_layer_norm,
    "Dropout": make_dropout,
    "Tanh": make_tanh,
}


def build_layers(init_shape: list[int], layers_config: list[dict]) -> list:
    shape = init_shape
    layers = []
    for config in layers_config:
        config = config.copy()
        tp = config.pop("type")
        if tp not in FACTORY_MAP:
            raise RuntimeError("Unknown layer type: " + tp)
        factory = FACTORY_MAP[tp]
        layer, shape = factory(shape, **config)
        layers.append(layer)
    return layers


class LSTM(nn.Module):
    def __init__(self, config: dict):
        super(LSTM, self).__init__()

        self.seq_len = config["seq_len"]
        input_size = config["input_size"]
        self.input_shape = [self.seq_len, input_size]
        self.layers = build_layers(self.input_shape, config["layers"])
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Module):
                self.add_module(f"_child_{i}", self.layers[i])

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        out = input_x
        for layer in self.layers:
            out = layer(out)
            if isinstance(out, tuple):
                out = out[0]
        return out[:, -1]


class LinearBNReLU(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )


class Block(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.layer1 = LinearBNReLU(dim, dim, dropout)
        self.layer2 = LinearBNReLU(dim, dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + residual
        return x


class Layer(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, num_blocks: int):
        mods = [LinearBNReLU(in_dim, out_dim, dropout)]
        for _ in range(num_blocks):
            mods.append(Block(out_dim, dropout))
        super().__init__(*mods)


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        dim = config["input_size"]
        dropout = config["dropout"]
        self.layers = nn.Sequential(
            Layer(dim, 512, dropout, 1),
            Layer(512, 256, dropout, 1),
            Layer(256, 128, dropout, 1),
            Layer(128, 64, dropout, 1),
            nn.Linear(64, 1),
        )

    def forward(self, alpha: Tensor) -> Tensor:
        """
        alpha: [batch, 1, dim]
        return: [batch]
        """
        bs, seq_len, dim = alpha.shape
        assert seq_len == 1
        alpha = alpha.reshape(bs, dim)
        out = self.layers(alpha)
        out = out.squeeze(dim=1)
        out = out.reshape(bs, 1)
        return out
