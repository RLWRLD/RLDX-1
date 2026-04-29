import json
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel

from rldx.configs.base_config import Config
from rldx.data.collator import BasicDataCollator
from rldx.data.dataset.factory import DatasetFactory
from rldx.data.interfaces import BaseProcessor
from rldx.utils.dist import rank_zero_print as _print


class ModelPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.data_collator = None

    def setup(self):
        pass

    def return_model(self):
        return self.model

    def return_dataset(self):
        return self.train_dataset, self.eval_dataset

    def return_collator(self):
        return self.data_collator

    def return_processor(self):
        return self.processor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class BasicPipeline(ModelPipeline):
    """A simple pipeline that works for diffusion and flowmatching-based models."""

    model_class: type[PreTrainedModel]
    processor_class: type[BaseProcessor]
    data_collator_class: type[BasicDataCollator] = BasicDataCollator

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        # Load model
        model = self.model_class(self.config.model)
        _print("\nModel Config: ", model.config)

        # unfreeze the model first
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _print(f"Total parameters: {total_params:,}")
        _print(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        return model

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        self.processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            statistics=None,  # This will be computed and set later.
            **self.config.model.processor_kwargs,
        )
        dataset_factory = DatasetFactory(self.config)
        train_dataset, eval_dataset = dataset_factory.build(self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        _print("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.data_collator_class()
        return data_collator
