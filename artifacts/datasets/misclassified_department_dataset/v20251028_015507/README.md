---
datasets:
- sambodhan/misclassified_department_dataset
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
pretty_name: Sambodhan Grievance Dataset (Department)
---

# Dataset: `sambodhan/misclassified_department_dataset`

Processed and versioned dataset for department classification.

---

## Version Information
- **Version Tag:** `v20251028_015507`
- **Created At:** 2025-10-28T01:55:10.901225+00:00
- **Label Column:** `department`
- **Total Samples:** 6000

### Label Mapping
| Label | ID |
|:------|:--:|
| Municipal Governance & Community Services | 0 |
| Education, Health & Social Welfare | 1 |
| Infrastructure, Utilities & Natural Resources | 2 |
| Security & Law Enforcement | 3 |

## Dataset Splits
- **Train**: 4800 samples
- **Eval**: 600 samples
- **Test**: 600 samples

## Task Description
This dataset contains preprocessed citizen grievance texts for classification tasks:
- Department classification

## Author
- **Maintainer:** `mr-kush`

## Pipeline Information
This dataset is automatically generated and versioned by the **Sambodhan AI Data Pipeline**.
It ensures:
- Continuous version tracking  
- Consistent preprocessing standards  
- Reproducibility for fine-tuning and evaluation  

---

_Last updated automatically by the pipeline on 2025-10-28T01:55:10.901225+00:00._
