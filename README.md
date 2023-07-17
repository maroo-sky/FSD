# Feature Structure Distillation with Centered Kernel Alignment for BERT Transferring
**ELSEVIER Expert Systems with Applications (accepted paper)**

## Abstract
Knowledge distillation is an approach to transfer information on representations from a teacher to a student by reducing their difference. A challenge of this approach is to reduce the flexibility of the student’s representations inducing inaccurate learning of the teacher’s knowledge. To resolve the problems, we propose a novel method feature structure distillation that elaborates information on structures of features into three types for transferring, and implements them based on Centered Kernel Analysis. In particular, the global local-inter structure is proposed to transfer the structure beyond the mini-batch. In detail, the method first divides the feature information into three structures: intra-feature, local inter-feature, and global inter-feature structures to subdivide the structure and transfer the diversity of the structure. Then, we adopt CKA which shows a more accurate similarity metric compared to other metrics between two different models or representations on different spaces. In particular, a memory-augmented transfer method with clustering is implemented for the global structures. The methods are empirically analyzed on the nine tasks for language understanding of the GLUE dataset with Bidirectional Encoder Representations from Transformers (BERT), which is a representative neural language model. In the results, the proposed methods effectively transfer the three types of structures and improves performance compared to state-of-the-art distillation methods: (i.e.) ours achieve 66.61% accuracy compared to the baseline (65.55%) in the RTE dataset. Indeed, the code for the methods is available at https://github.com/maroo-sky/FSD.

## Overview

<img src="./src/figures/intra-and%20local%20inter-feature.png" width="350" height="600" /> <img src="./src/figures/global%20structure.png" width="450" height="600" />

## Installaiton

##### Create virtual environment

    conda create -n "ENV_NAME" numpy=1.18.1 pandas=1.0.1 scikit-learn=0.22.1 pytorch=1.4.0 tqdm=4.42.1 python=3.7.6 matplotlib=3.1.3 

##### After creating virtual environment download belows
    
    pip install transformers==2.6.0 tensorboard==2.2.1

##### Then, change the sentencepiece library version
    
    pip install sentencepiece==0.1.91
    
##### If you got under ERROR message

    if this call came from a _pb2.py file your generated code is out of date and must be regenerated with protoc >-3.19.0.

##### Following under command

    pip install protobuf==3.20.*

## Training

**Objective Functions**

FSD_I: L = L_{KD} + \beta * L_{I}

FSD_L: L = L+{KD} + \beta * L_{L}

FSD_G: L = L_{KD} + L_{G}

FSD_{ILG}: L = L_{KD} + L_{ILG}

Please see our paper for more details.

## Dataset

We conduct on the GLUE benchmark and download in [GLUE official page](https://gluebenchmark.com/tasks)


## Fine-tuning Teacher model

Follow [Hugging Face](https://huggingface.co/transformers/v2.6.0/examples.html) to fine-tuning on GLUE benchmark.
 
we use "bert-base-uncased" and different learning rate, we present in our paper 

**Output directory must be followed under formation**
    
    output_dir=/your/path/to/bert-base-uncased

### Feature Structure Distillation

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

### Articles in press
[https://doi.org/10.1016/j.eswa.2023.120980](https://doi.org/10.1016/j.eswa.2023.120980)

### Reference (Bitex format)
    @article{JUNG2023120980,
    title = {Feature structure distillation with Centered Kernel Alignment in BERT transferring},
    journal = {Expert Systems with Applications},
    pages = {120980},
    year = {2023},
    issn = {0957-4174},
    doi = {https://doi.org/10.1016/j.eswa.2023.120980},
    url = {https://www.sciencedirect.com/science/article/pii/S0957417423014823},
    author = {Hee-Jun Jung and Doyeon Kim and Seung-Hoon Na and Kangil Kim},
    keywords = {Knowledge distillation, BERT, Centered Kernel Alignment, Natural language processing},
    abstract = {Knowledge distillation is an approach to transfer information on representations from a teacher to a student by reducing their difference. A challenge of this approach is to reduce the flexibility of the student’s representations inducing inaccurate learning of the teacher’s knowledge. To resolve the problems, we propose a novel method feature structure distillation that elaborates information on structures of features into three types for transferring, and implements them based on Centered Kernel Analysis. In particular, the global local-inter structure is proposed to transfer the structure beyond the mini-batch. In detail, the method first divides the feature information into three structures: intra-feature, local inter-feature, and global inter-feature structures to subdivide the structure and transfer the diversity of the structure. Then, we adopt CKA which shows a more accurate similarity metric compared to other metrics between two different models or representations on different spaces. In particular, a memory-augmented transfer method with clustering is implemented for the global structures. The methods are empirically analyzed on the nine tasks for language understanding of the GLUE dataset with Bidirectional Encoder Representations from Transformers (BERT), which is a representative neural language model. In the results, the proposed methods effectively transfer the three types of structures and improves performance compared to state-of-the-art distillation methods: (i.e.) ours achieve 66.61% accuracy compared to the baseline (65.55%) in the RTE dataset. Indeed, the code for the methods is available at https://github.com/maroo-sky/FSD.}
    }




    
    
    
 
    



