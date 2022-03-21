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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from model.bert_kd import BertForSequenceClassification, BertStudentMemoryForSequenceClassification

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    #BertForSequenceClassification,
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
from model.memory import Memory
#add libraries
import pdb
import collections
from model_config.bert_kd_config import StudentBertConfig, KDBertConfig
from metrics.LayerWiseMetrics import *
import torch.nn.functional as F
from metrics.CosEumap import *
from metrics.estimate_metrics import estimate_metrics
#from metrics.restoration_ratio import restoration
from src.write_results_to_csv import write_results
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

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, teacher, teacher_mem, student, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.run_file)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.student_model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.student_model_name_or_path, "scheduler.pt")
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
        teacher = torch.nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
        teacher_mem = torch.nn.DataParallel(teacher_mem)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
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
    tr_dark_loss, logging_dark_loss = 0.0, 0.0
    tr_global_loss, logging_global_loss = 0.0, 0.0
    tr_dist_struc_loss, logging_dist_struc_loss = 0.0, 0.0
    tr_cos_struc_loss, logging_cos_struc_loss = 0.0, 0.0

    tr_pkd, logging_pkd = 0.0, 0.0
    tr_inter_dist, logging_inter_dist = 0.0, 0.0
    tr_inter_cos, logging_inter_cos = 0.0, 0.0
    tr_intra_dist, logging_intra_dist = 0.0, 0.0
    tr_intra_cos, logging_intra_cos = 0.0, 0.0

    # teacher, student distance mean & std
    tr_inter_TD_mean, logging_inter_TD_mean = 0.0, 0.0
    tr_inter_TD_var, logging_inter_TD_var = 0.0, 0.0
    tr_inter_SD_mean, logging_inter_SD_mean = 0.0, 0.0
    tr_inter_SD_var, logging_inter_SD_var = 0.0, 0.0

    tr_intra_TD_mean, logging_intra_TD_mean = 0.0, 0.0
    tr_intra_TD_var, logging_intra_TD_var = 0.0, 0.0
    tr_intra_SD_mean, logging_intra_SD_mean = 0.0, 0.0
    tr_intra_SD_var, logging_intra_SD_var = 0.0, 0.0

    # teacher, student cosine mean & std

    tr_inter_TC_mean, logging_inter_TC_mean = 0.0, 0.0
    tr_inter_TC_var, logging_inter_TC_var = 0.0, 0.0
    tr_inter_SC_mean, logging_inter_SC_mean = 0.0, 0.0
    tr_inter_SC_var, logging_inter_SC_var = 0.0, 0.0

    tr_intra_TC_mean, logging_intra_TC_mean = 0.0, 0.0
    tr_intra_TC_var, logging_intra_TC_var = 0.0, 0.0
    tr_intra_SC_mean, logging_intra_SC_mean = 0.0, 0.0
    tr_intra_SC_var, logging_intra_SC_var = 0.0, 0.0

    student.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
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
            #student_outputs : ce loss, logit, hidden, centroid, ckaloss

            kld_loss = nn.KLDivLoss(reduction='batchmean')
            l2_loss = nn.MSELoss()
            #teacher_featuremaps, student_featuremaps = collections.OrderedDict(), collections.OrderedDict()

            #teacher_featuremaps[0] = teacher_outputs[2][12]
            #student_featuremaps[0] = student_outputs[2][6]

            #length = len(teacher_featuremaps)

            loss = student_outputs[0]  # model outputs are always tuple in transformers (see doc)
            teacher_logit, student_logit = teacher_outputs[1], student_outputs[1]
            teacher_feature = teacher_outputs[2][11].view(teacher_outputs[2][11].size()[0], -1 )
            student_memory = student_outputs[3]
            if args.n_gpu > 1:
                teacher_memory = teacher_mem.module.centroid
            else:
                teacher_memory = teacher_mem.centroid
            norm_stu_memory = F.normalize(student_memory, dim=-1)
            norm_tea_memory = F.normalize(teacher_memory, dim=-1)
            norm_teacher_feature = F.normalize(teacher_feature, dim=-1)

            #stu_memory_dist_structure = cdist2(student_memory, student_memory)
            #tea_memory_dist_structure = cdist2(teacher_memory, teacher_memory)
            #stu_memory_cos_structure = torch.mm(norm_stu_memory, norm_stu_memory.transpose(-1, -2))
            #tea_memory_cos_structure = torch.mm(norm_tea_memory, norm_tea_memory.transpose(-1, -2))
            tea_dist_structure = cdist2(teacher_feature, teacher_memory).squeeze()
            tea_cos_structure = torch.mm(norm_teacher_feature ,norm_tea_memory.transpose(-1,-2))
            stu_dist_structure, stu_cos_structure = student_outputs[4], student_outputs[5]
            #memory_dist_loss = l2_loss(stu_memory_dist_structure, tea_memory_dist_structure)
            #memory_cos_loss = l2_loss(stu_memory_cos_structure, tea_memory_cos_structure)
            memory_cka_loss = linear_CKA_loss(student_memory, teacher_memory, args)
            dist_structure_loss = l2_loss(stu_dist_structure, tea_dist_structure)
            cos_structure_loss = l2_loss(stu_cos_structure, tea_cos_structure)

            dark_knowledge = kld_loss(F.log_softmax(student_logit.unsqueeze(1) / args.temperature, dim=-1),
                                        F.softmax(teacher_logit.unsqueeze(1) / args.temperature,
                                                  dim=-1)) * args.temperature * args.temperature

            #student_memory_hidden_loss = student_outputs[4]
            #teacher_memory_hidden_loss = linear_CKA_loss(teacher_memory.centroid, teacher_feature, args)
            #memory_loss = linear_CKA_loss(teacher_memory.centroid, student_memory, args)

            global_loss = args.gamma * (dist_structure_loss) + (1 - args.gamma) * (cos_structure_loss) \
                          + memory_cka_loss
            #+memory_dist_loss +memory_cos_loss

            total_loss = args.alpha * loss + (1 - args.alpha) * dark_knowledge + args.beta * global_loss

            # make featuremaps dict for estimate all losses.
            teacher_featuremaps, student_featuremaps = collections.OrderedDict(), collections.OrderedDict()
            for i in range(5):
                teacher_featuremaps[i] = teacher_outputs[2][2 * (i + 1)]
                student_featuremaps[i] = student_outputs[2][i + 1]
            teacher_featuremaps[-1] = teacher_outputs[2][11]
            student_featuremaps[-1] = student_outputs[2][5]

            pkd, dist_inter, cos_inter, dist_intra, cos_intra = estimate_metrics(teacher_featuremaps,
                                                                                 student_featuremaps, args)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                total_loss = total_loss.mean()
                dark_knowledge = dark_knowledge.mean()
                cka_loss = cka_loss.mean()
                # before change : exter, inter --> inter, intra
                estimate_pkd = estimate_pkd.mean()
                for i in range(5):
                    dist_inter[i] = dist_inter[i].mean()
                    dist_intra[i] = dist_intra[i].mean()
                    cos_inter[i] = cos_inter[i].mean()
                    cos_intra[i] = cos_intra[i].mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            tr_loss += loss.item()
            tr_total_loss += total_loss.item()
            tr_global_loss += global_loss.item()
            tr_dark_loss += dark_knowledge.item()
            # temporal variable to check loss
            tr_dist_struc_loss += dist_structure_loss.item()
            tr_cos_struc_loss += cos_structure_loss.item()

            tr_pkd += pkd.item()
            tr_inter_dist += dist_inter[0].item()
            tr_intra_dist += dist_intra[0].item()
            tr_inter_cos += cos_inter[0].item()
            tr_intra_cos += cos_intra[0].item()

            tr_inter_TD_mean += dist_inter[1].item()
            tr_intra_TD_mean += dist_intra[1].item()
            tr_inter_TC_mean += cos_inter[1].item()
            tr_intra_TC_mean += cos_intra[1].item()

            tr_inter_TD_var += dist_inter[2].item()
            tr_intra_TD_var += dist_intra[2].item()
            tr_inter_TC_var += cos_inter[2].item()
            tr_intra_TC_var += cos_intra[2].item()

            tr_inter_SD_mean += dist_inter[3].item()
            tr_intra_SD_mean += dist_intra[3].item()
            tr_inter_SC_mean += cos_inter[3].item()
            tr_intra_SC_mean += cos_intra[3].item()

            tr_inter_SD_var += dist_inter[4].item()
            tr_intra_SD_var += dist_intra[4].item()
            tr_inter_SC_var += cos_inter[4].item()
            tr_intra_SC_var += cos_intra[4].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                student.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well

                        results = evaluate(args, student, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    tb_writer.add_scalar("00.cross_entrophy", (tr_loss - logging_loss) / args.logging_steps,
                                         global_step)
                    tb_writer.add_scalar("01.lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("02.total_loss", (tr_total_loss - logging_total_loss) / args.logging_steps,
                                         global_step)
                    tb_writer.add_scalar("03.logit_loss", (tr_dark_loss - logging_dark_loss) / args.logging_steps,
                                         global_step)
                    tb_writer.add_scalar("04.Global_loss",
                                         (tr_global_loss - logging_global_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar("05.PKD_loss",
                                         (tr_pkd - logging_pkd) / args.logging_steps, global_step)
                    tb_writer.add_scalar("06.Inter_Distance",
                                         (tr_inter_dist - logging_inter_dist) / args.logging_steps, global_step)
                    tb_writer.add_scalar("07.Inter_Cosine",
                                         (tr_inter_cos - logging_inter_cos) / args.logging_steps, global_step)
                    tb_writer.add_scalar("08.Intra_Distance",
                                         (tr_intra_dist - logging_intra_dist) / args.logging_steps, global_step)
                    tb_writer.add_scalar("09.Intra_Cosine",
                                         (tr_intra_cos - logging_intra_cos) / args.logging_steps, global_step)

                    tb_writer.add_scalar("10.Inter_Distance_T_mean",
                                         (tr_inter_TD_mean - logging_inter_TD_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("11.Inter_Distance_S_mean",
                                         (tr_inter_SD_mean - logging_inter_SD_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("12.Inter_Cosine_T_mean",
                                         (tr_inter_TC_mean - logging_inter_TC_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("13.Inter_Cosine_S_mean",
                                         (tr_inter_SC_mean - logging_inter_SC_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("14.Intra_Distance_T_mean",
                                         (tr_intra_TD_mean - logging_intra_TD_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("15.Intra_Distance_S_mean",
                                         (tr_intra_SD_mean - logging_intra_SD_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("16.Intra_Cosine_T_mean",
                                         (tr_intra_TC_mean - logging_intra_TC_mean) / args.logging_steps, global_step)
                    tb_writer.add_scalar("17.Intra_Cosine_S_mean",
                                         (tr_intra_SC_mean - logging_intra_SC_mean) / args.logging_steps, global_step)

                    tb_writer.add_scalar("18.Inter_Distance_T_var",
                                         (tr_inter_TD_var - logging_inter_TD_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("19.Inter_Distance_S_var",
                                         (tr_inter_SD_var - logging_inter_SD_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("20.Inter_Cosine_T_var",
                                         (tr_inter_TC_var - logging_inter_TC_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("21.Inter_Cosine_S_var",
                                         (tr_inter_SC_var - logging_inter_SC_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("22.Intra_Distance_T_var",
                                         (tr_intra_TD_var - logging_intra_TD_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("23.Intra_Distance_S_var",
                                         (tr_intra_SD_var - logging_intra_SD_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("24.Intra_Cosine_T_var",
                                         (tr_intra_TC_var - logging_intra_TC_var) / args.logging_steps, global_step)
                    tb_writer.add_scalar("25.Intra_Cosine_S_var",
                                         (tr_intra_SC_var - logging_intra_SC_var) / args.logging_steps, global_step)
                    #temporal variable to check loss
                    tb_writer.add_scalar("26.Dist_Structure_loss",
                                        tr_dist_struc_loss - logging_dist_struc_loss / args.logging_steps, global_step)
                    tb_writer.add_scalar("27.Cos_Structure_loss",
                                         tr_cos_struc_loss - logging_cos_struc_loss / args.logging_steps, global_step)

                    logging_loss = tr_loss
                    logging_total_loss = tr_total_loss
                    logging_dark_loss = tr_dark_loss
                    logging_global_loss = tr_global_loss
                    # temporal variable to check loss
                    logging_dist_struc_loss = tr_dist_struc_loss
                    logging_cos_struc_loss = tr_cos_struc_loss

                    logging_pkd = tr_pkd
                    logging_inter_dist = tr_inter_dist
                    logging_inter_cos = tr_inter_cos
                    logging_intra_dist = tr_intra_dist
                    logging_intra_cos = tr_intra_cos

                    logging_inter_TD_mean = tr_inter_TD_mean
                    logging_intra_TD_mean = tr_intra_TD_mean
                    logging_inter_TC_mean = tr_inter_TC_mean
                    logging_intra_TC_mean = tr_intra_TC_mean

                    logging_inter_TD_var = tr_inter_TD_var
                    logging_intra_TD_var = tr_intra_TD_var
                    logging_inter_TC_var = tr_inter_TC_var
                    logging_intra_TC_var = tr_intra_TC_var

                    logging_inter_SD_mean = tr_inter_SD_mean
                    logging_intra_SD_mean = tr_intra_SD_mean
                    logging_inter_SC_mean = tr_inter_SC_mean
                    logging_intra_SC_mean = tr_intra_SC_mean

                    logging_inter_SD_var = tr_inter_SD_var
                    logging_intra_SD_var = tr_intra_SD_var
                    logging_inter_SC_var = tr_inter_SC_var
                    logging_intra_SC_var = tr_intra_SC_var

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        student.module if hasattr(student, "module") else student
                    )  # Take care of distributed/parallel training
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

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
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
        file_dir = args.output_dir.split(os.sep)[:-1]
        file_dir = os.path.join('/', *file_dir, 'result.csv')
        write_results(filename=file_dir, args=args, results=result)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def mem_rep_tsne(args, teacher, teacher_memory, model, tokenizer, prefix=""):
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
        student_reps, teacher_reps = None, None
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
                student_outputs = model(**inputs)
                teacher_outputs = teacher(**inputs)
                student_rep = student_outputs[2][5]
                teacher_rep = teacher_outputs[2][11]

            if student_reps is None:
                student_reps = student_rep.detach().cpu().numpy()
                teacher_reps = teacher_rep.detach().cpu().numpy()
            else:
                student_reps = np.append(student_rep, student_rep.detach().cpu().numpy(), axis=0)
                teacher_reps = np.append(teacher_rep, student_rep.detach().cpu().numpy(), axis=0)

        student_memory = student_outputs[2].detach().cpu().numpy()
        teacher_memory = teacher_memory.centroid if args.n_gpu > 1 else teacher_memory.module.centroid
        teacher_memory = teacher_memory.detach().cpu().numpy()
        pdb.set_trace()

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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
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
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
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
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    #add parser
    parser.add_argument(
        "--model_name_or_path_for_copy",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument("--student_model_name_or_path",
                        default=None,
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in teh list: " + ", ".join(
                            ALL_MODELS)),
    parser.add_argument("--student_model_type",
                        default=None,
                        type=str,
                        help="Model type selected in the list: " + ", ".join(STUDENT_MODEL_CLASSES.keys()),
                        )
    parser.add_argument("--teacher_memory_path",
                        default=None,
                        type=str,
                        help="Path to teacher memory")
    parser.add_argument("--copy_teacher", action="store_true", help="Copy teacher's layers parameter")

    #parser.add_argument("--do_layerwise", action="store_true", help="Layer-wise KD.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Hyper parameter of hard label.")
    parser.add_argument("--beta", default=30.0, type=float, help="Hyper parameter of cka loss.")
    parser.add_argument("--gamma", default=0.5, type=float, help="Hyper parameter of global loss")
    parser.add_argument("--temperature", type=float, default=5.0, help="Determine softmax temperature")
    parser.add_argument("--run_file", type=str, required=True, help="create tensorboard run files")
    #parser.add_argument("--internal", action="store_true", help="internal CKA")
    parser.add_argument("--num_centroid", type=int, default=100, required=False, help="choose num of centroid")
    parser.add_argument("--internal", action="store_true", help="CKA for internal structure")
    parser.add_argument("--do_tsne", action="store_true")
    #arser.add_argument("--memory_num_train_epochs", type=int, required=True, help="num of memory training")
    #parser.add_argument("--memory_learning_rate", type=float, required=True, help="")

    args = parser.parse_args()

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

    # Set seed
    set_seed(args)

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
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES[args.model_type]
    student_config_class, student_class, student_tokenizer_class = STUDENT_MODEL_CLASSES[args.student_model_type]

    teacher_config = teacher_config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
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

    student_config = student_config_class(
            num_labels=num_labels,
            finetuning_task=args.task_name,
            num_centroid=args.num_centroid,
            max_seq_length=args.max_seq_length,
            )

    if args.copy_teacher:
        student_model = student_class.from_pretrained(
            args.model_name_or_path_for_copy,
            from_tf=bool("ckpt" in args.model_name_or_path_for_copy),
            config=student_config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        student_model = student_class.from_pretrained(config=student_config)
    teacher_memory = torch.load(args.teacher_memory_path)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    teacher_model.to(args.device)
    student_model.to(args.device)
    teacher_memory.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    sub_path = "alpha:" + str(args.alpha) + "_beta:" + str(args.beta) + "_gamma_memory:" + str(args.gamma) \
               + "_lr:" + str(args.learning_rate) + "_temperature:" + str(args.temperature) + "_epoch:" + str(
        args.num_train_epochs) \
               + "_seed:" + str(args.seed)
    args.output_dir = os.path.join(args.output_dir, sub_path)
    args.run_file = os.path.join(args.run_file, sub_path)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, teacher_model, teacher_memory, student_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            student_model.module if hasattr(student_model, "module") else student_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = student_model.from_pretrained(args.output_dir)
        tokenizer = teacher_tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = teacher_tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = student_model.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_tsne:
        mem_rep_tsne(args, teacher_model, teacher_memory, student_model, tokenizer)
    return results


if __name__ == "__main__":
    main()
