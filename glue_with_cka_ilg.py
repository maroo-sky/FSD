# intra-, inter- and global feature sturcture distillation

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from model.bert_kd import BertStudentMemoryForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers import glue_compute_metrics as compute_metrics
from dataLoader.glue import glue_convert_examples_to_features as convert_examples_to_features
from dataLoader.glue import glue_output_modes as output_modes
from dataLoader.glue import glue_processors as processors

import collections
from model_config.bert_kd_config import StudentBertConfig, KDBertConfig
from metrics.LayerWiseMetrics import *
import torch.nn.functional as F
from metrics.CosEumap import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (KDBertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}
STUDENT_MODEL_CLASSES = {
        "bert" : (StudentBertConfig, BertStudentMemoryForSequenceClassification, BertTokenizer ),
}

def set_seed(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, teacher, teacher_mem, student, tokenizer, alpha, beta, gamma, sigma, temperature, num_train_epochs, learning_rate, seed):
    """ Train the model """
    set_seed(args,seed)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.student_model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.student_model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.student_model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        student, optimizer = amp.initialize(student, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)
        teacher_mem = torch.nn.DataParallel(teacher_mem)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        teacher_mem = torch.nn.DataParallel(teacher_mem)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.student_model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    tr_total_loss, logging_total_loss = 0.0, 0.0
    tr_logit_loss, logging_logit_loss = 0.0, 0.0

    student.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args, seed)

    # Create dictionary for EU-Cos map
    for i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            teacher.eval()
            teacher_mem.eval()
            student.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            teacher_outputs = teacher(**inputs)
            student_outputs = student(**inputs)
            cka = CKA_mixed_loss(args)
            kld_loss = nn.KLDivLoss(reduction='batchmean')
            l2_loss = nn.MSELoss()

            loss = student_outputs[0]  # model outputs are always tuple in transformers (see doc)
            teacher_logit, student_logit = teacher_outputs[1], student_outputs[1]
            teacher_feature = teacher_outputs[2][11].view(teacher_outputs[2][11].size()[0], -1)
            teacher_featuremaps, student_featuremaps = collections.OrderedDict(), collections.OrderedDict()
            if args.do_layerwise:
                for i in range(5):
                    teacher_featuremaps[i] = teacher_outputs[2][2 * (i + 1)]
                    student_featuremaps[i] = student_outputs[2][i + 1]
            else:
                teacher_featuremaps[0] = teacher_outputs[2][11]
                student_featuremaps[0] = student_outputs[2][5]

            length = len(teacher_featuremaps)
            cka_inter_loss, cka_exter_loss = cka(teacher_featuremaps, student_featuremaps, args, length)

            student_memory = student_outputs[3]
            teacher_memory = teacher_mem.centroid
            #norm_stu_memory = F.normalize(student_memory, dim=-1)
            norm_tea_memory = F.normalize(teacher_memory, dim=-1)
            norm_teacher_feature = F.normalize(teacher_feature, dim=-1)

            #stu_memory_dist_structure = torch.cdist(student_memory, student_memory)
            #tea_memory_dist_structure = torch.cdist(teacher_memory, teacher_memory)
            #stu_memory_cos_structure = torch.mm(norm_stu_memory, norm_stu_memory.transpose(-1, -2))
            #tea_memory_cos_structure = torch.mm(norm_tea_memory, norm_tea_memory.transpose(-1, -2))
            tea_dist_structure = torch.cdist(teacher_feature, teacher_memory)
            tea_cos_structure = torch.mm(norm_teacher_feature, norm_tea_memory.transpose(-1, -2))
            stu_dist_structure, stu_cos_structure = student_outputs[4], student_outputs[5]

            memory_cka_loss = linear_CKA_loss(student_memory, teacher_memory, args)
            #memory_dist_loss = l2_loss(stu_memory_dist_structure, tea_memory_dist_structure)
            #memory_cos_loss = l2_loss(stu_memory_cos_structure, tea_memory_cos_structure)
            dist_structure_loss = l2_loss(stu_dist_structure, tea_dist_structure)
            cos_structure_loss = l2_loss(stu_cos_structure, tea_cos_structure)

            dark_knowledge = kld_loss(F.log_softmax(student_logit.unsqueeze(1) / temperature, dim=-1),
                                      F.softmax(teacher_logit.unsqueeze(1) / temperature,
                                                dim=-1)) * temperature * temperature

            # student_memory_hidden_loss = student_outputs[4]
            # teacher_memory_hidden_loss = linear_CKA_loss(teacher_memory.centroid, teacher_feature, args)
            # memory_loss = linear_CKA_loss(teacher_memory.centroid, student_memory, args)

            #global_loss = beta * (memory_dist_loss + dist_structure_loss) + \
            #              (1 - beta) * (memory_cos_loss + cos_structure_loss) #beta * memory_cka_loss \
            global_loss = beta * (dist_structure_loss) + (1 - beta) * (cos_structure_loss) + memory_cka_loss

            total_loss = alpha * loss + (1 - alpha) * dark_knowledge + args.epsilon * (sigma * global_loss + \
                         (1 - sigma) * 0.5 * (gamma * cka_inter_loss + (1 - gamma) * cka_exter_loss))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                total_loss = total_loss.mean()
                dark_knowledge = dark_knowledge.mean()

            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            tr_loss += loss.item()
            tr_total_loss += total_loss.item()
            tr_logit_loss += dark_knowledge.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                student.zero_grad()
                global_step += 1

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = student.module if hasattr(student, "module") else student
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_total_loss / global_step


#def evaluate(args, model, tokenizer, prefix="", validation=False):
def evaluate(args, model, tokenizer, prefix=""):
    #dataset, examples, features = load_and_cache_examples(args, tokenizer, validation, evaluate=True, output_examples=True )
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)


    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def write_results(filename, alpha, beta, gamma, sigma,  temp, epoch, lr, batch, seed, result, args):

    file_exists = os.path.isfile(filename)

    info = {}
    fieldnames = []
    info["alpha"], info["beta"], info["gamma"], info["sigma"], info["epsilon"], info["temp"], info["epoch"], info["lr"], info["batch"], info["seed"] =\
        alpha, beta, gamma, sigma, args.epsilon, temp, epoch, lr, seed, batch

    for key in sorted(result.keys()):
        info[key] = result[key]

    for key in info:
        fieldnames.append(str(key))

    with open(filename, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(info)

    return

def grid_search(args):

    origin_path = args.output_dir

    for alpha in args.alpha:
        for temperature in args.temperature:
            for epoch in args.num_train_epochs:
                for beta in args.beta:
                    for gamma in args.gamma:
                        for sigma in args.sigma:
                            for lr in args.learning_rate:
                                for seed in args.seed:

                                    # Set seed
                                    set_seed(args, seed)

                                    # Prepare GLUE task
                                    args.task_name = args.task_name.lower()
                                    if args.task_name not in processors:
                                        raise ValueError("Task not found: %s" % (args.task_name))
                                    processor = processors[args.task_name]()
                                    args.output_mode = output_modes[args.task_name]
                                    label_list = processor.get_labels()
                                    num_labels = len(label_list)

                                    # Load pretrained model and tokenizer
                                    if args.local_rank not in [-1, 0]:
                                        # Make sure only the first process in distributed training will download model & vocab
                                        torch.distributed.barrier()

                                    args.model_type = args.model_type.lower()
                                    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES[
                                        args.model_type]
                                    student_config_class, student_model_class, student_tokenizer_class = STUDENT_MODEL_CLASSES[
                                        args.student_model_type]

                                    teacher_config = teacher_config_class.from_pretrained(
                                        args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                    )
                                    tokenizer = teacher_tokenizer_class.from_pretrained(
                                        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                        do_lower_case=args.do_lower_case,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                    )
                                    teacher_model = teacher_model_class.from_pretrained(
                                        args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=teacher_config,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                    )

                                    teacher_memory = torch.load(args.teacher_memory_path)
                                    student_config = student_config_class(num_labels=num_labels,
                                                                          finetuning_task=args.task_name,
                                                                          num_centroid=args.num_centroid,
                                                                          max_seq_length=args.max_seq_length,)

                                    if args.copy_teacher:
                                        student_model = student_model_class.from_pretrained(
                                            args.model_name_or_path_for_copy,
                                            from_tf=bool("ckpt" in args.model_name_or_path_for_copy),
                                            config=student_config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,
                                        )
                                    else:
                                        student_model = student_model_class(config=student_config)

                                    if args.local_rank == 0:
                                        # Make sure only the first process in distributed training will download model & vocab
                                        torch.distributed.barrier()

                                    teacher_model.to(args.device)
                                    student_model.to(args.device)
                                    teacher_memory.to(args.device)
                                    args.output_dir = os.path.join(args.output_dir,
                                                                   "pkd_" + str(args.task_name) + "_alpha:" + str(alpha) + "_beta:" + str(beta) + "_gamma:" + str(gamma) +
                                                                   "_sigma:" + str(sigma) + "_epsilon:" + str(args.epsilon) + \
                                                                   "_temperature:" + str(temperature) + \
                                                                        "_epoch:" + str(epoch) + "_lr:" + str(lr) + "seed:" + str(seed),\
                                                                   'bert-base-uncased')

                                    #train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
                                    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
                                    global_step, tr_loss = train(args, train_dataset, teacher_model, teacher_memory, student_model,
                                                                 tokenizer, alpha, beta, gamma, sigma, temperature, epoch, lr, seed)

                                    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                                        #Create output directory if needed
                                        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                                            os.makedirs(args.output_dir)

                                        logger.info("Saving model checkpoint to %s", args.output_dir)
                                        #Svae a trained model, configuration and tokenizer using save_pretrained()
                                        #They can then be reloaded using 'from_pretrained()'
                                        #Take care of distributed.parallel training
                                        model_to_save = student_model.module if hasattr(student_model, "module") else student_model
                                        model_to_save.save_pretrained(args.output_dir)
                                        tokenizer.save_pretrained(args.output_dir)

                                        # Good practice: save your training arguments together with the trained model
                                        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                                        #Load a trained model and vocabulary that you have fine-tuned
                                        model = student_model_class.from_pretrained(args.output_dir) #, force_download=True
                                        model.to(args.device)

                                    if args.do_eval and args.local_rank in [-1, 0]:
                                        if args.do_train:
                                            logger.info("Loading checkpints saved during training for evaluation")
                                            checkpoints = [args.output_dir]
                                            if args.eval_all_checkpoints:
                                                checkpoints = list(
                                                    os.path.dirname(c)
                                                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                                                )
                                                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
                                        else:
                                            logger.info("Loading checkpoint %s for evaluation", args.student_model_name_or_path)
                                            checkpoints = [args.student_model_name_or_path]

                                        logger.info("Evaluate the following checkpoints: %s", checkpoints)

                                        for checkpoint in checkpoints:
                                            global_step = checkpoints.split("-")[-1] if len(checkpoints) > 1 else""
                                            model = student_model_class.from_pretrained(checkpoint)
                                            model.to(args.device)
                                            results = evaluate(args, model, tokenizer)

                                            subdir = args.output_dir
                                            subdir = subdir.split(os.path.sep)
                                            subdir = subdir[:-2]
                                            subdir = os.path.join('/',*subdir)

                                            filename = os.path.join(subdir, "result.csv")
                                            write_results(filename=filename,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          gamma=gamma,
                                                          sigma=sigma,
                                                          temp=temperature,
                                                          epoch=epoch,
                                                          lr=lr,
                                                          batch=args.per_gpu_train_batch_size,
                                                          seed=args.seed,
                                                          result=results,
                                                          args=args)

                                    args.output_dir = origin_path


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    #add parsers
    parser.add_argument("--do_hyper", action="store_true", help="Hyper parameter tuning.")
    parser.add_argument("--student_model_name_or_path",
                        default=None,
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in teh list: " + ", ".join(ALL_MODELS)),
    #parser.add_argument("--check_pkd", action="store_true", help="Check pkd similarity.")
    #parser.add_argument("--check_pkd", action="store_true", help="Check PKD distance.")
    parser.add_argument("--do_mapping", action="store_true", help="Shows relationship between distance and cosine similarity.")
    parser.add_argument("--do_layerwise", action="store_true", help="Layer-wise KD.")
    parser.add_argument("--do_normalize", action="store_true", help="Normalize for pkd")
    parser.add_argument("--student_model_type",
                        default=None,
                        type=str,
                        help="Model type selected in the list: " +", ".join(STUDENT_MODEL_CLASSES.keys()),
                        )
    parser.add_argument("--copy_teacher", action="store_true", help="Copy teacher's layers parameter")
    parser.add_argument("--validation", action="store_true", help="Devide train and validation dataset")
    #parser.add_argument("--run_file", type=str, required=True, help="create tensorboard run files")

    parser.add_argument("--permutation", type=float, default=0.3, help="ratio of train dataset to valid dataset")
    parser.add_argument("--alpha", type=float, nargs='*', default=[0.0], help="range of alpha for grid search")
    parser.add_argument("--beta", type=float, nargs='*', default=[5.0], help="range of beta for grid search")
    parser.add_argument("--gamma", type=float, nargs='*', default=[0.0], help="range of alpha for grid search")
    parser.add_argument("--sigma", type=float, nargs='*', default=[5.0], help="range of beta for grid search")
    parser.add_argument("--epsilon", type=float, default=0.1, help="range of epsilon of epsilon for grid search")
    parser.add_argument("--temperature", type=float, nargs='*', default=[3])
    parser.add_argument("--learning_rate",  type=float, nargs='*', default=[5e-5], help="The initial learning rate for Adam.")
    parser.add_argument("--seed", type=int, nargs='*', default=[42], help="random seed for initialization")
    parser.add_argument(
        "--num_train_epochs", type=float, nargs='*', default=[3.0], help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--model_name_or_path_for_copy",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument("--num_centroid", type=int, default=100, required=False, help="choose num of centroid")
    parser.add_argument("--teacher_memory_path",
                        default=None,
                        type=str,
                        help="Path to teacher memory")

    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )




    grid_search(args)



    if args.do_mapping:
        featuremap_checkpoints = torch.load(args.featuremap_checkpoint_dir)
        sim_mat, dist_mat = tensor_dictionary_to_numpy(featuremap_checkpoints, batch=args.per_gpu_train_batch_size,
                                                   num=args.extract_point, epoch=args.num_train_epochs)
        save_CosineSimilarity_Distance(sim_mat, dist_mat, args.save_figure_dir)



if __name__ == "__main__":
    main()