# Knowledge Distillation by Transferring Intra-and Inter-Feature Structure in Transformer

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


##Fine-tuing Teacher model

Follow [Hugging Face](https://huggingface.co/transformers/v2.6.0/examples.html) to fine-tuning on GLUE benchmark.
 
we use "bert-base-uncased" and different learning rate, we present in our paper 

Output directory must be followed under formation
    
    output_dir=/your/path/to/bert-base-uncased

### Feature Strucuture Distillation

**Intra-and Local Inter-Feature Distillation Method (FSD_I & FSD_L)**

    DATA_DIR = /{path_of_data}/
    OUTPUT_DIR=/{path_of_fine-tuned_model}/bert-base-uncased #be a student model
    MODEL_PATH = /{path_of_fine-tuned_teacher_model}/bert-base-uncased # teacher model
    RUN_FILE_PATH = /{path_for_tensorboard_fun_file}/
    TASK_NAME = glue_task

    python run_glue_intra_inter.py --model_type bert \
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
    
    python run_glue_global.py --model_type bert 
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
    
    python run_glue_all.py --model_type bert 
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

    DATA_DIR = /{path_of_data}/
    OUTPUT_DIR=/{path_of_fine-tuned_model}/bert-base-uncased #be a student model
    MODEL_PATH = /{path_of_fine-tuned_teacher_model}/bert-base-uncased # teacher model
    RUN_FILE_PATH = /{path_for_tensorboard_fun_file}/
    TASK_NAME = glue_task
    
    python train_memory.py --model_type bert \
    --model_name_or_path MODEL_PATH  \
    --data_dir DATA_DIR \
    --per_gpu_train_batch_size 32 \
    --max_seq_length 128 \
    --output_dir OUTPUT_DIR\
    --do_train \
    --logging_steps 500000 \
    --save_steps 500000 \
    --run_file RUN_FILE_PATH \
    --num_train_epochs 3 \
    --memory_num_train_epochs 3 \
    --task_name TASK_NAME \
    --do_lower_case \
    --num_centroid 100 \
    


    
    
    
 
    



