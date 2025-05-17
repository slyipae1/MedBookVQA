#!/bin/bash

workspace="" # path to this repo

model_path=API/gemini-2.5-pro-preview-03-25
python eval.py --workspace=$workspace --model_path=$model_path

model_path=API/gpt-4o-2024-11-20
python eval.py --workspace=$workspace --model_path=$model_path

model_path=API/claude-3-7-sonnet-20250219
python eval.py --workspace=$workspace --model_path=$model_path

model_path=InternVL3-1B
python eval.py --workspace=$workspace --model_path=$model_path

model_path=llava-onevision-qwen2-0.5b-ov # llava-onevision-qwen2-72b-ov-chat; llava-onevision-qwen2-7b-ov-chat
python eval.py --workspace=$workspace --model_path=$model_path

model_path=InternVL2_5-1B
python eval.py --workspace=$workspace --model_path=$model_path

model_path=Ovis2-1B
python models/Ovis.py --workspace=$workspace --model_path=$model_path

model_path=Qwen2.5-VL-3B-Instruct
python eval.py --workspace=$workspace --model_path=$model_path

model_path=deepseek-vl2-tiny
python eval.py --workspace=$workspace --model_path=$model_path

model_path=HuatuoGPT-Vision-7B # HuatuoGPT-Vision-34B
python eval.py --workspace=$workspace --model_path=$model_path


############################################################################################################
############################################################################################################
# reasoning 

model_path=API/claude-3-7-sonnet-thinking
python eval.py --workspace=$workspace --model_path=$model_path --reasoning

model_path=MedVLM-R1 
python eval.py --workspace=$workspace --model_path=$model_path --reasoning

model_path=/data/Skywork-R1V2-38B # Skywork-R1V-38B
python eval.py --workspace=$workspace --model_path=$model_path--reasoning

model_path=VL-Reasoner-7B # VL-Reasoner-72B
python eval.py --workspace=$workspace --model_path=$model_path --reasoning

model_path=Kimi-VL-A3B-Thinking
python eval.py --workspace=$workspace --model_path=$model_path --reasoning

if parse result only
python eval.py --workspace=$workspace --model_path=$model_path --parse_reasoning --reasoning


############################################################################################
###################################################################################
# special cases

cd evaluation/models/HealthGPTmodel/llava/DOABbench_test/
model_path=HealthGPT-M3
python eval.py --workspace=$workspace --model_path=$model_path
model_path=HealthGPT-L14
python eval.py --workspace=$workspace --model_path=$model_path