---
datasets:
- sambodhan/misclassified_urgency_dataset
language:
- en
- ne
task_categories:
- text-classification
task_ids:
- multi-class-classification
license: apache-2.0
size_categories:
- 1K<n<10K
pretty_name: Sambodhan Grievance Dataset (Urgency)
---

# Dataset: `sambodhan/misclassified_urgency_dataset`

Processed and versioned dataset for urgency classification.

---

## Version Information
- **Version Tag:** `v20251030_111553`
- **Created At:** 2025-10-30T11:16:03.891837+00:00
- **Label Column:** `urgency`
- **Total Samples:** 1500

### Label Mapping
| Label | ID |
|:------|:--:|
| NORMAL | 0 |
| URGENT | 1 |
| HIGHLY URGENT | 2 |

## Dataset Splits
- **Train**: 1200 samples
- **Eval**: 150 samples
- **Test**: 150 samples

## Task Description
This dataset contains preprocessed citizen grievance texts for classification tasks:
- Urgency classification

## Author
- **Maintainer:** `mr-kush`

## Pipeline Information
This dataset is automatically generated and versioned by the **Sambodhan AI Data Pipeline**.
It ensures:
- Continuous version tracking  
- Consistent preprocessing standards  
- Reproducibility for fine-tuning and evaluation  

---

_Last updated automatically by the pipeline on 2025-10-30T11:16:03.891837+00:00._
