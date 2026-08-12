---
language:
- ne
- en
license: apache-2.0
tags:
- text-classification
- xlm-roberta
- multilingual
- complaint-classification
- urgency-detection
- transformers
datasets:
- custom
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: sambodhan-urgency-classifier
  results:
  - task:
      type: text-classification
      name: Complaint Urgency Classification
    metrics:
    - type: accuracy
      value: 0.9467
      name: Accuracy
    - type: f1
      value: 0.9475
      name: F1 Macro
    - type: f1
      value: 0.9464
      name: F1 Weighted
base_model:
- FacebookAI/xlm-roberta-base
---

# Sambodhan Urgency Classifier

## Model Description

This model classifies complaint/grievance texts into urgency levels using XLM-RoBERTa base. It's trained to identify:
- **NORMAL (0)**: Standard complaints requiring regular processing
- **URGENT (1)**: Time-sensitive complaints needing prompt attention  
- **HIGHLY URGENT (2)**: Critical complaints requiring immediate action

The model supports multilingual inputs (Nepali and English) and is optimized for small datasets using advanced techniques like Focal Loss and class weighting.

## Model Details

- **Base Model**: xlm-roberta-base
- **Task**: Multi-class Text Classification (3 classes)
- **Languages**: Nepali (ne), English (en)
- **Training Dataset Size**: ~5781 samples
- **Max Sequence Length**: 96 tokens

## Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.9467 |
| F1 Macro | 0.9475 |
| F1 Weighted | 0.9464 |

### Per-Class Performance

| Class | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| NORMAL | 0.9296 | 0.9472 | 0.9127 |
| URGENT | 0.9410 | 0.9449 | 0.9372 |
| HIGHLY URGENT | 0.9718 | 0.9485 | 0.9961 |

## Usage

```python
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

# Load model and tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained("YOUR_USERNAME/sambodhan-urgency-classifier")
tokenizer = XLMRobertaTokenizer.from_pretrained("YOUR_USERNAME/sambodhan-urgency-classifier")

# Prepare input
text = "बिजुली काटिएको छ र कुनै सूचना छैन"  # Example in Nepali
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

# Map to label
id2label = {0: "NORMAL", 1: "URGENT", 2: "HIGHLY URGENT"}
print(f"Predicted urgency: {id2label[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class].item():.2%}")
```

## Training Details

### Training Hyperparameters

- **Learning Rate**: 3e-05
- **Batch Size**: 16 (with gradient accumulation: 2)
- **Epochs**: 15
- **Weight Decay**: 0.02
- **Warmup Ratio**: 0.15
- **Dropout**: 0.3
- **Max Length**: 96

### Training Results

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Macro | F1 Weighted |
|:------:|:--------------:|:----------------:|:----------:|:----------:|:-------------:|
| 1 | 0.600400 | 0.564277 | 0.373333 | 0.293781 | 0.266230 |
| 2 | 0.261500 | 0.216027 | 0.736667 | 0.730709 | 0.714058 |
| 3 | 0.183200 | 0.264484 | 0.863333 | 0.866542 | 0.863632 |
| 4 | 0.100000 | 0.111478 | 0.924444 | 0.926013 | 0.924094 |
| 5 | 0.081800 | 0.156611 | 0.935556 | 0.936081 | 0.935261 |
| 6 | 0.080800 | 0.094597 | 0.942222 | 0.943049 | 0.941953 |
| 7 | 0.062600 | 0.098532 | 0.941111 | 0.942484 | 0.941026 |
| 8 | 0.067600 | 0.092606 | 0.935556 | 0.936776 | 0.935182 |
| 9 | 0.050700 | 0.121297 | 0.943333 | 0.943942 | 0.942912 |
| 10 | 0.045000 | 0.091817 | 0.946667 | 0.947465 | 0.946381 |
| 11 | 0.037000 | 0.106251 | 0.940000 | 0.941134 | 0.939696 |
| 12 | 0.024600 | 0.115572 | 0.940000 | 0.941328 | 0.939698 |
| 13 | 0.039200 | 0.127271 | 0.941111 | 0.942034 | 0.940779 |
| 14 | 0.024500 | 0.120282 | 0.943333 | 0.944187 | 0.943002 |


### Advanced Techniques Used

1. **Focal Loss** (γ=2.0): Focuses on hard-to-classify examples
2. **Class Weighting**: Balanced training with adjusted weights for minority classes
3. **Data Augmentation**: Word dropout, swap, and duplication techniques
4. **Label Smoothing**: Prevents overconfident predictions
5. **Cosine Learning Rate Schedule**: With warmup for stable training

## Limitations and Biases

- The model's performance is limited by the size and quality of the training dataset
- May have difficulty with domain-specific jargon or very short texts
- Performance may vary for code-mixed text (Nepali-English)
- Class imbalance in training data may affect predictions

## Citation

If you use this model, please cite:

```
@misc{sambodhan-urgency-classifier,
  author = {Karan Bista},
  title = {Sambodhan Urgency Classifier},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/YOUR_USERNAME/sambodhan-urgency-classifier}}
}
```

## Model Card Authors

Created by: Karan Bista
Contact: bistakaran89@gmail.com

## Model Card Contact

For questions or feedback, please open an issue in the model repository or contact bistakaran89@gmail.com.