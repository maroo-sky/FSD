# FSD: Feature Structure Distillation
This work has been submitted to the IEEE for possible publication. IEEE Transitions of Neural Networks and Learning Systems (TNNLS)

## Abstract
Knowledge distillation is an approach to transfer information on representations from a teacher to a student by reducing their difference. 
A challenge of this approach is to reduce the flexibility of the student's representations inducing inaccurate learning of the teacher's knowledge. 
To resolve it in BERT transferring, we investigate distillation of structures of representations specified to three types: intra-feature, 
local inter-feature, global inter-feature structures. 
To transfer them, we introduce *feature structure distillation* methods based on the Centered Kernel Alignment, 
which assigns a consistent value to similar features structures and reveals more informative relations. 
In particular, a memory-augmented transfer method with clustering is implemented for the global structures. 
In the experiments on the nine tasks for language understanding of the GLUE dataset, 
the proposed methods effectively transfer the three types of structures and improve performance compared to state-of-the-art distillation methods. 
Indeed, the code for the methods is available in this https URL.

## Overview

<img src="./src/figures/intra-and%20local%20inter-feature.png" width=350" height="600" /> <img src="./src/figures/global%20structure.png" width="450" height="600" />

## Installaiton

##### Create virture environment

    conda create -n "ENV_NAME" numpy=1.18.1 pandas=1.0.1 scikit-learn=0.22.1 pytorch=1.4.0 tqdm=4.42.1 python=3.7.6 matplotlib=3.1.3 

##### After creating virture environment download belows
    
    pip install transformers==2.6.0 tensorboard==2.2.1

##### Then, change the sentencepiece library version
    
    pip install sentencepiece==0.1.91

## Training

**Objective Functions**

FSD_I: L = L_{KD} + \beta * L_{I}

FSD_L: L = L+{KD} + \beta * L_{L}

FSD_G: L = L_{KD} + L_{G}

FSD_{ILG}: L = L_{KD} + L_{ILG}

Pleas see our paper for more details.

## Dataset

We contuct on the GLUE benchmark and download in [GLUE official page](https://gluebenchmark.com/tasks)


## Fine-tuing Teacher model

Follow [Hugging Face](https://huggingface.co/transformers/v2.6.0/examples.html) to fine-tuning on GLUE benchmark.
 
we use "bert-base-uncased" and different learning rate, we present in our paper 

**Output directory must be followed under formation**
    
    output_dir=/your/path/to/bert-base-uncased

### Feature Strucuture Distillation

**Intra-and Local Inter-Feature Distillation Method (FSD_I & FSD_L)**
    
run "glue_with_cka_l.py" for CKA_{inter}. without "--internal"
    
    DATA_DIR = /{path_of_data}/
    OUTPUT_DIR=/{path_of_fine-tuned_model}/bert-base-uncased #be a student model
    MODEL_PATH = /{path_of_fine-tuned_teacher_model}/bert-base-uncased # teacher model
    RUN_FILE_PATH = /{path_for_tensorboard_fun_file}/
    TASK_NAME = glue_task

    python glue_with_cka_i.py --model_type bert \
    --model_name_or_path MODEL_PATH  \
    --model_name_or_path_for_copy bert-base-uncased \
    --data_dir DATA_DIR \
    --per_gpu_train_batch_size 32 \
    --max_seq_length 128 \
    --output_dir OUTPUT_DIR \
    --student_model_type bert \
    --student_model_name_or_path OUTPUT_DIR \
    --do_eval \
    --do_train \
    --run_file RUN_FILE_PATH\
    --copy_teacher \
    --alpha alpha \
    --beta beta \
    --temperature temperature \
    --num_train_epochs num_train_epoch \
    --learning_rate learning_rate \
    --task_name TASK_NAME\
    --do_lower_case\
    --internal 
    
    optional to train methods
    #--copy_teacher: copy fine-tuned teacher model parameters
    #--internal: do Intra-Feature Distillation

**Global Inter-Feature Distillation Method (FSD_G)**
    
    DATA_DIR = /{path_of_data}/
    OUTPUT_DIR=/{path_of_fine-tuned_model}/bert-base-uncased #be a student model
    MODEL_PATH = /{path_of_fine-tuned_teacher_model}/bert-base-uncased # teacher model
    RUN_FILE_PATH = /{path_for_tensorboard_fun_file}/
    TASK_NAME = glue_task
    MEMORY_PATH = /{path_of_trained_memory}/
    
    python glue_with_cka_g.py --model_type bert 
    --model_name_or_path MODEL_PATH  
    --model_name_or_path_for_copy bert-base-uncased 
    --data_dir DATA_DIR 
    --per_gpu_train_batch_size 32 
    --max_seq_length 128 
    --output_dir OUTPUT_DIR 
    --overwrite_output_dir 
    --student_model_type bert 
    --student_model_name_or_path OUTPUT_DIR 
    --do_eval --do_train 
    --run_file RUN_FILE_PATH 
    --copy_teacher 
    --alpha alpha
    --beta beta
    --gamma_m gamm_m
    --temperature tmeperature 
    --num_train_epochs num_train_epochs
    --learning_rate learning_rate
    --task_name wnli --do_lower_case 
    --teacher_memory_path MEMORY_PATH
    
    optional to train with outher methods
    
**Integrated Distillation Methods (FSD_{ILG})**
    
    DATA_DIR = /{path_of_data}/
    OUTPUT_DIR=/{path_of_fine-tuned_model}/bert-base-uncased #be a student model
    MODEL_PATH = /{path_of_fine-tuned_teacher_model}/bert-base-uncased # teacher model
    RUN_FILE_PATH = /{path_for_tensorboard_fun_file}/
    TASK_NAME = glue_task
    MEMORY_PATH = /{path_of_trained_memory}/
    
    python glue_with_cka_ilg.py --model_type bert 
    --model_name_or_path MODEL_PATH  
    --model_name_or_path_for_copy bert-base-uncased 
    --data_dir DATA_DIR 
    --per_gpu_train_batch_size 32 
    --max_seq_length 128 
    --output_dir OUTPUT_DIR 
    --overwrite_output_dir 
    --student_model_type bert 
    --student_model_name_or_path OUTPUT_DIR 
    --do_eval --do_train 
    --run_file RUN_FILE_PATH 
    --copy_teacher 
    --alpha alpha
    --beta beta
    --gamma_m gamm_m
    --gamma_i gamma_i
    --gamma_l gamma_l
    --gamma_g gamma_g
    --temperature tmeperature 
    --num_train_epochs num_train_epochs
    --learning_rate learning_rate
    --task_name wnli --do_lower_case 
    --teacher_memory_path MEMORY_PATH
    
    optional to train with outher methods
    gamma_g = 0.0 for FSD_{IL}

### Train Teacher Memory

**Global Inter-Feature Distillation**

num_centroid is 300 on MNLI, QQP dataset, except it is 100.

    DATA_DIR = /{path_of_data}/
    OUTPUT_DIR=/{path_of_fine-tuned_model}/bert-base-uncased #be a student model
    MODEL_PATH = /{path_of_fine-tuned_teacher_model}/bert-base-uncased # teacher model
    RUN_FILE_PATH = /{path_for_tensorboard_fun_file}/
    TASK_NAME = glue_task upper case (ex: WNLI)
    task = glue_task lower case (ex: wnli)
    
    python train_memory.py --model_type bert \
    --model_name_or_path MODEL_PATH  \
    --data_dir DATA_DIR \
    --per_gpu_train_batch_size 32 \
    --max_seq_length 128 \
    --output_dir OUTPUT_DIR\
    --do_train \
    --logging_steps 500000 \
    --save_steps 500000 \
    --num_train_epochs 3 \
    --memory_num_train_epochs 3 \
    --task_name TASK_NAME \
    --do_lower_case \
    --num_centroid 100 \
    --memory_learning_rate \
    --task_name $task

### Pre-print version
[https://arxiv.org/abs/2204.08922](https://arxiv.org/abs/2204.08922)

### Reference (Bitex format)
    @misc{https://doi.org/10.48550/arxiv.2204.08922,
      doi = {10.48550/ARXIV.2204.08922},
      url = {https://arxiv.org/abs/2204.08922},
      author = {Jung, Hee-Jun and Kim, Doyeon and Na, Seung-Hoon and Kim, Kangil},
      keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Feature Structure Distillation for BERT Transferring},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}
    }



    
    
    
 
    



