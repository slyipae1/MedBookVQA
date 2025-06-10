
<div align="center">

# MedBookVQA:  Systematic and Comprehensive Medical Benchmark Derived from Open-Access Book

<div align="center">

</div>

[\[🤗 Dataset\]](https://huggingface.co/datasets/slyipae1/MedBookVQA/) 
[\[📜 Paper\]](https://arxiv.org/abs/2506.00855)

</div>


## Introduction
The rapid rise of general medical AI (GMAI), powered by multimodal large language models (MLLMs), offers promising solutions to healthcare challenges like workforce shortages and rising costs. Systematic benchmarks are essential to evaluate and guide these technologies. Medical textbooks, though rich in structured knowledge, remain underutilized for this purpose.

We introduce **MedBookVQA**, a multimodal benchmark built from open-access medical textbooks. MedBookVQA contains 5,000 clinically meaningful visual-question-answering (VQA) tasks spanning five clinical task types: modality recognition, disease classification, anatomical identification, symptom diagnosis, and surgical procedures. To facilitate detailed performance analysis, the benchmark is organized by a hierarchical annotation system covering 42 imaging modalities, 125 anatomical structures, and 31 clinical specialties. 

MedBookVQA highlights the value of textbook-based benchmarks for advancing clinical AI and provides structured insights into current model limitations across medical domains.

+ MedBookVQA Benchmark Results
![MedBookVQA Benchmark Results](assets/AllResult.png)



## Usage
### Data
+ Download the data from [Hugging Face](https://huggingface.co/datasets/slyipae1/MedBookVQA/) to this folder, and unzip figures.zip. Organize files as follows:
    + evaluation
    + figures
    + data.json
        
    <details>
    <summary>Meta of data.json</summary>

    ```
    "_id": {
        "type": "string",
        "description": "Unique identifier for each entry."
        },
        "QAtype": {
        "type": "string",
        "enum": [
            "Modality Recognition",
            "Disease Classification",
            "Anatomy Identification",
            "Symptom Diagnosis",
            "Surgery & Operation"
        ],
        "description": "Type of the question asked."
        },
        "Question": {
        "type": "string",
        "description": "The question posed regarding the medical image."
        },
        "Answer": {
        "type": "string",
        "description": "The correct answer to the question."
        },
        "Distractors": {
        "type": "array",
        "items": {
            "type": "string"
        },
        "description": "List of distractor answers."
        },
        "correct_choice": {
        "type": "string",
        "description": "The correct answer choice (e.g., 'C'), corresponds to Answer."
        },
        "other_choices": {
        "type": "array",
        "items": {
            "type": "string"
        },
        "description": "List of other answer choices, corresponds to Distractors."
        },
        "image_path": {
        "type": "string",
        "description": "Path to the associated medical image."
        },
        "LABEL": {
        "type": "dict",
        "properties": {
            "Modality": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Hierarchical labels for modality (ModalityCategory, Modality)."
            },
            "Anatomy": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Hierarchical labels for anatomy (System, SubSystem, BodyPart, Organ)."
            },
            "Department": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Hierarchical labels for department (DepartmentCategory, Department)."
            }
        }
        }
    ```

    </details>

    <details>
    <summary>Sample entry of data.json</summary>

    ```
    {
    "_id": "c115303a242b0d28140ad4f50903c63d",
    "QAtype": "Anatomy Identification",
    "Question": " What anatomical structure is prominently visible in the center of the mammogram? ",
    "Answer": " Lactiferous sinus zone",
    "Distractors": [
      "Mammary ductal system",
      "Fibroglandular tissue area",
      "Areolar complex region"
    ],
    "correct_choice": "C",
    "other_choices": [
      "A",
      "B",
      "D"
    ],
    "image_path": "./figures/c115303a242b0d28140ad4f50903c63d.jpg",
    "LABEL": {
      "Modality": [
        "Electrical Impedance Tomography",
        "Electrical Impedance Tomography"
      ],
      "Anatomy": [
        "Integumentary",
        "Integumentary",
        "Mammary gland",
        "Mammary gland"
      ],
      "Department": [
        "Obstetrics and Gynecology",
        "Obstetrics and Gynecology"
      ]
    }
    }
    ```

    </details>


## Evaluation
+ Change the image paths using "evaluation/change_path.py" if needed.
+ Run "evaluation/eval.py" to evaluate. (Env: Please refer to setup of the corresponding model.)
+ Run "evaluation/eval.py" to evaluate. (Env: Please refer to setup of the corresponding model.)
+ Use "evaluation/summerize_result.ipynb" for analysis


## Citation
```
@misc{yip2025medbookvqasystematiccomprehensivemedical,
      title={MedBookVQA: A Systematic and Comprehensive Medical Benchmark Derived from Open-Access Book}, 
      author={Sau Lai Yip and Sunan He and Yuxiang Nie and Shu Pui Chan and Yilin Ye and Sum Ying Lam and Hao Chen},
      year={2025},
      eprint={2506.00855},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.00855}, 
}
```