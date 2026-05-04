from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.model.abstract_components import TorchPredictorModel
from src.model.composite_model import CompositeModel
from src.model.embedders import ESM


class FakeTokenizer:
    mask_token = "<mask>"

    def batch_encode_plus(self, seqs, add_special_tokens=True, padding="longest"):
        lengths = [len(seq.split()) + 2 for seq in seqs]
        max_len = max(lengths)
        return {
            "input_ids": [[1] * length + [0] * (max_len - length) for length in lengths],
            "attention_mask": [[1] * length + [0] * (max_len - length) for length in lengths],
        }


class FakeModel:
    config = SimpleNamespace(num_hidden_layers=2)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        shape = (*input_ids.shape, 3)
        values = torch.arange(np.prod(shape), dtype=torch.float, device=input_ids.device).reshape(
            shape
        )
        if output_hidden_states:
            return SimpleNamespace(hidden_states=[values, values + 100, values + 200])
        return SimpleNamespace(last_hidden_state=values)


def make_esm(mean_pool=False, output_hidden_states=False, layer_idx=None):
    embedder = ESM.__new__(ESM)
    embedder.cfg = OmegaConf.create(
        {
            "embedder": {
                "mean_pool": mean_pool,
                "n_copies": 1,
                "output_hidden_states": output_hidden_states,
                "layer_idx": layer_idx,
                "mask_placeholder": None,
            }
        }
    )
    embedder.device = torch.device("cpu")
    embedder.tokenizer = FakeTokenizer()
    embedder.model = FakeModel()
    return embedder


def test_process_batch_defaults_to_cpu_tensors():
    embeddings = make_esm().process_batch(["ACD", "EF"])

    assert [embedding.device.type for embedding in embeddings] == ["cpu", "cpu"]
    assert [embedding.shape for embedding in embeddings] == [
        torch.Size([3, 3]),
        torch.Size([2, 3]),
    ]


def test_process_batch_uses_requested_device():
    embeddings = make_esm().process_batch(["ACD"], return_device=torch.device("cpu"))

    assert embeddings[0].device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_process_batch_can_return_cuda_tensors():
    embedder = make_esm()
    embedder.device = torch.device("cuda")

    embeddings = embedder.process_batch(["ACD"], return_device=torch.device("cuda"))

    assert embeddings[0].device.type == "cuda"


def test_mean_pooled_forward_stays_numpy_compatible():
    embeddings = make_esm(mean_pool=True).forward(["ACD"], return_device=torch.device("cpu"))

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 3)


class DummyTorchPredictor(TorchPredictorModel):
    def forward(self):
        pass

    def train_model(self):
        pass

    def save_model(self):
        pass

    def get_hparams_string(self):
        pass

    def get_param_grid(self):
        pass


class RecordingEmbedder:
    def __init__(self):
        self.return_device = "not-called"

    def forward(self, sequences, return_device=None):
        self.return_device = return_device
        return [torch.ones(2, 3)]


def make_composite(run_mode="test", mean_pool=False, predictor=None):
    model = CompositeModel.__new__(CompositeModel)
    model.cfg = OmegaConf.create(
        {
            "general": {"run_mode": run_mode},
            "embedder": {"class_name": "ESM", "mean_pool": mean_pool, "standardize": False},
        }
    )
    model.embedder = RecordingEmbedder()
    model.predictor = predictor if predictor is not None else object()
    return model


def test_composite_requests_cuda_for_safe_torch_inference(monkeypatch):
    predictor = DummyTorchPredictor(
        OmegaConf.create(
            {
                "predictor": {
                    "model_type": "classification_binary",
                    "hparams": {"batch_size": 1, "max_epochs": 1, "patience": 1},
                }
            }
        )
    )
    model = make_composite(predictor=predictor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model.embed_sequences(["ACD"])

    assert model.embedder.return_device == torch.device("cuda")


@pytest.mark.parametrize(
    ("run_mode", "mean_pool", "predictor"),
    [
        ("train", False, None),
        ("test", True, None),
        ("test", False, object()),
    ],
)
def test_composite_keeps_cpu_default_outside_safe_torch_inference(
    monkeypatch,
    run_mode,
    mean_pool,
    predictor,
):
    model = make_composite(run_mode=run_mode, mean_pool=mean_pool, predictor=predictor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model.embed_sequences(["ACD"])

    assert model.embedder.return_device is None
