import os, sys
sys.path.append(os.getcwd())
from os.path import join, exists
import pathlib
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset, DistributedSampler
from PIL import Image
import torch
import torch.distributed as dist
import itertools

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')

import transformers

from configs.unified_config import ModelArguments, DataArguments, TrainingArguments, InferenceArguments
from dataset.unified_dataset import get_dataset_collator
from utils.util import set_seed, find_all_linear_names, prepare_sample, write2json, load_ckpt
from utils.deepspeed_utils import *


def setup_distributed():
    """Initialize distributed environment"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def inference_avqa(dataset, collator, dataloader, ckpt_dir, model, tokenizer, num):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    save_dir = join(ckpt_dir, 'inference_avqa')
    os.makedirs(save_dir, exist_ok=True)

    fp = join(save_dir, f'results_rank{rank}.jsonl')

    subset_dataloader = dataloader
    pbar = tqdm(total=len(subset_dataloader), desc=f'[Rank {rank}] Inference', disable=(rank != 0))

    for step, sample in enumerate(subset_dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data=sample)
        sample.update({'use_cache': True, 'max_new_tokens': 150})

        with torch.no_grad():
            output = model.generate(**sample)
            output = tokenizer.batch_decode(output, skip_special_tokens=False)

        for i in range(bs):
            metadata = batch_metadata[i]
            metadata['predict'] = output[i]
            write2json(fp=fp, dict_data=metadata)

        pbar.update(1)

    pbar.close()
    dist.barrier()
    if rank == 0:
        print("All ranks finished inference.")

def train(attn_implementation=None):
    set_seed(42)

    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()

    if model_args.llm_name == 'llama':
        d_model = 4096
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig

        pretrain_model_name_or_path = model_args.model_name_or_path
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation

        compute_dtype = torch.float32
        if training_args.fp16:
            compute_dtype = torch.float16
        elif training_args.bf16:
            compute_dtype = torch.bfloat16

        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            torch_dtype=compute_dtype
        )

    model.config.use_cache = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft_hyper import LoraConfig, get_peft_model
        lora_trainable = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
        target_modules = lora_trainable.split(',')
        lora_rank = training_args.lora_r
        lora_alpha = 16
        lora_dropout = 0.05
        lora_nums = int(len(str(training_args.lora_r)))
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank,
            loramethod=training_args.loramethod,
            reserved_modality=training_args.reserved_modality,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_nums=lora_nums,
        )
        model = get_peft_model(model, peft_config)

    if model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.get_model().init_multimodal_modules(
        visual_branch=training_args.visual_branch,
        audio_branch=training_args.audio_branch,
        d_model=4096,
        vit_ckpt_path=model_args.vit_ckpt_path,
        select_layer_list=model_args.select_layer_list,
        select_feature=model_args.select_feature,
        image_size=model_args.image_size,
        patch_size=model_args.patch_size,
        visual_query_token_nums=model_args.visual_query_token_nums,
        audio_query_token_nums=model_args.audio_query_token_nums,
        BEATs_ckpt_path=model_args.BEATs_ckpt_path
    )

    model.initialize_MM_tokenizer(tokenizer)
    MM_tokenizer_vocab_nums = len(tokenizer)
    if rank == 0:
        print(f'ori_tokenizer_vocab_nums: {ori_tokenizer_vocab_nums}, MM_tokenizer_vocab_nums: {MM_tokenizer_vocab_nums}')

    ckpt_dir = infer_args.ckpt_dir
    ckpt_path = join(ckpt_dir, 'finetune_weights.bin')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    if rank == 0:
        print(f'load ckpt from {ckpt_path} finished...')

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model.eval()

    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(
        data_args=data_args, tokenizer=tokenizer,
        image_processor=image_processor, mode='test'
    )

    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, collate_fn=collator, drop_last=False)

    inference_avqa(dataset=dataset, collator=collator, dataloader=dataloader,
                   ckpt_dir=ckpt_dir, model=model, tokenizer=tokenizer, num=infer_args.cut_folds)

    dist.barrier()
    if rank == 0:
        print("Inference complete across all GPUs.")

if __name__ == "__main__":
    train()
