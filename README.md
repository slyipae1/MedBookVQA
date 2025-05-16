# MedBookVQA

## Introduction
The accelerating development of general medical artificial intelligence (GMAI), powered by multimodal large language models (MLLMs), presents transformative potential to tackle persistent healthcare challenges, including workforce shortages and rising costs. To evaluate these advancements effectively, the establishment of systematic benchmarks is essential.

Introducing **MedBookVQA**, a systematic and comprehensive multimodal benchmark derived from open-access medical textbooks. Our approach involves a standardized pipeline for the automated extraction of medical figures, aligned with relevant narratives. We generate 5,000 clinically relevant questions covering modality recognition, disease classification, anatomical identification, symptom diagnosis, and surgical procedures.

With a multi-tier annotation system categorizing queries across 42 imaging modalities, 125 anatomical structures, and 31 clinical specialties, MedBookVQA enables nuanced performance analysis in various medical subdomains. Our evaluation of diverse MLLMs uncovers significant performance disparities, emphasizing critical gaps in current GMAI systems and establishing textbook-derived benchmarks as vital tools for advancing clinical AI.


## Usage
### Data
+ Create a folder "workspace"
+ Download the data from [Hugging Face](https://huggingface.co/datasets/slyipae1/MedBookVQA/) to this folder

## Evaluation
+ Change the image path using "evaluation/change_path.py" if needed.
+ Run "evaluation/eval.py"
+ Use "evaluation/summerize_result.ipynb" for analysis