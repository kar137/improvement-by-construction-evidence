---
library_name: transformers
license: mit
base_model: xlm-roberta-base
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: sambodhan_urgency_classifier
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sambodhan_urgency_classifier

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0687
- Accuracy: 0.6867
- F1 Macro: 0.6909
- F1 Weighted: 0.6900
- Precision Macro: 0.7047
- Recall Macro: 0.6878
- Precision Weighted: 0.7039
- Recall Weighted: 0.6867

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
| 0.123         | 0.6667 | 50   | 0.1217          | 0.3333   | 0.1974   | 0.2000      | 0.2353          | 0.3276       | 0.2350             | 0.3333          |
| 0.1191        | 1.3333 | 100  | 0.0996          | 0.5333   | 0.5322   | 0.5309      | 0.5440          | 0.5353       | 0.5433             | 0.5333          |
| 0.0918        | 2.0    | 150  | 0.0881          | 0.5933   | 0.5835   | 0.5821      | 0.6286          | 0.5960       | 0.6280             | 0.5933          |
| 0.0674        | 2.6667 | 200  | 0.0687          | 0.6867   | 0.6909   | 0.6900      | 0.7047          | 0.6878       | 0.7039             | 0.6867          |


### Framework versions

- Transformers 4.57.1
- Pytorch 2.9.0+cu128
- Datasets 4.3.0
- Tokenizers 0.22.1
