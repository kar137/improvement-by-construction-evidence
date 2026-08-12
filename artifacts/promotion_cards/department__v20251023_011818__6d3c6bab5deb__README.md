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

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0209
- Accuracy: 0.9417
- F1 Macro: 0.9417
- F1 Weighted: 0.9419
- Precision Macro: 0.9375
- Recall Macro: 0.9480
- Precision Weighted: 0.9445
- Recall Weighted: 0.9417

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
| 0.1451        | 0.1667 | 50   | 0.0692          | 0.8067   | 0.7985   | 0.8017      | 0.8190          | 0.8135       | 0.8288             | 0.8067          |
| 0.0424        | 0.3333 | 100  | 0.0181          | 0.9583   | 0.9567   | 0.9581      | 0.9614          | 0.9533       | 0.9590             | 0.9583          |
| 0.0219        | 0.5    | 150  | 0.0209          | 0.9417   | 0.9417   | 0.9419      | 0.9375          | 0.9480       | 0.9445             | 0.9417          |


### Framework versions

- Transformers 4.57.1
- Pytorch 2.8.0+cu126
- Datasets 4.0.0
- Tokenizers 0.22.1
