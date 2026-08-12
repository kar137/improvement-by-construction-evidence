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
- Loss: 0.0156
- Accuracy: 0.9667
- F1 Macro: 0.9655
- F1 Weighted: 0.9666
- Precision Macro: 0.9671
- Recall Macro: 0.9640
- Precision Weighted: 0.9668
- Recall Weighted: 0.9667

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
| 0.1763        | 0.1667 | 50   | 0.1054          | 0.7133   | 0.5942   | 0.6345      | 0.5342          | 0.6877       | 0.5849             | 0.7133          |
| 0.065         | 0.3333 | 100  | 0.0232          | 0.9417   | 0.9389   | 0.9421      | 0.9387          | 0.9414       | 0.9448             | 0.9417          |
| 0.0253        | 0.5    | 150  | 0.0235          | 0.945    | 0.9449   | 0.9451      | 0.9407          | 0.9503       | 0.9465             | 0.945           |
| 0.0133        | 0.6667 | 200  | 0.0187          | 0.965    | 0.9642   | 0.9651      | 0.9619          | 0.9668       | 0.9655             | 0.965           |
| 0.0126        | 0.8333 | 250  | 0.0158          | 0.97     | 0.9685   | 0.9702      | 0.9677          | 0.9696       | 0.9706             | 0.97            |
| 0.0109        | 1.0    | 300  | 0.0156          | 0.9667   | 0.9655   | 0.9666      | 0.9671          | 0.9640       | 0.9668             | 0.9667          |


### Framework versions

- Transformers 4.57.1
- Pytorch 2.8.0+cu126
- Datasets 4.0.0
- Tokenizers 0.22.1
