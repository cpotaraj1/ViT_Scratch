# ViT_Scratch
Building vision transformer from scratch!
Based on https://arxiv.org/abs/2010.11929

## Usage

Dependencies:
- PyTorch 1.13.1 ([install instructions](https://pytorch.org/get-started/locally/))
- torchvision 0.14.1 ([install instructions](https://pytorch.org/get-started/locally/))
- matplotlib 3.7.1 to generate plots for model inspection

The main class is `ViTForImageClassification` in utils/model_architecture.py, which contains the embedding layer, the transformer encoder, and the classification head. 

The model config is defined as a python dictionary in `train.py`, you can experiment with different hyperparameters there (More room to improvement by adding parser module to capture this parameters from command line).

```python
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
```

The model is much smaller than the original ViT models from the paper (which has at least 12 layers and hidden size of 768) as I just want to illustrate how the model works rather than achieving state-of-the-art performance.

There are other functions in utils/experiment.py to visualize the predictions and attention masks 