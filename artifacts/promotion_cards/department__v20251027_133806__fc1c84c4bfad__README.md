---
library_name: transformers
license: mit
base_model: xlm-roberta-base
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: sambodhan-department-classification-model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sambodhan-department-classification-model

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0008
- Accuracy: 0.9959
- F1 Macro: 0.9959
- F1 Weighted: 0.9959
- Precision Macro: 0.9959
- Recall Macro: 0.9959
- Precision Weighted: 0.9960
- Recall Weighted: 0.9959

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 32
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Accuracy | F1 Macro | F1 Weighted | Precision Macro | Recall Macro | Precision Weighted | Recall Weighted |
|:-------------:|:------:|:----:|:---------------:|:--------:|:--------:|:-----------:|:---------------:|:------------:|:------------------:|:---------------:|
| 0.1695        | 0.4098 | 50   | 0.0976          | 0.8436   | 0.8464   | 0.8466      | 0.8561          | 0.8436       | 0.8567             | 0.8436          |
| 0.042         | 0.8197 | 100  | 0.0021          | 0.9959   | 0.9959   | 0.9959      | 0.9959          | 0.9959       | 0.9960             | 0.9959          |
| 0.0046        | 1.2295 | 150  | 0.0001          | 1.0      | 1.0      | 1.0         | 1.0             | 1.0          | 1.0                | 1.0             |
| 0.0012        | 1.6393 | 200  | 0.0008          | 0.9959   | 0.9959   | 0.9959      | 0.9959          | 0.9959       | 0.9960             | 0.9959          |


### Framework versions

- Transformers 4.57.1
- Pytorch 2.9.0+cu128
- Datasets 4.3.0
- Tokenizers 0.22.1
