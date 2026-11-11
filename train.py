from data import get_dataloader
from utils.load import load_yaml
from utils.parse_arg import parse_argument
from utils.training_monitor import TrainingMonitor
from model import GPT2_RoPE
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
import os
import random
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# wandb 프로젝트 설정
wandb_log = False
wandb_project = "GPT2-DDP-Optimization"
wandb_run_name = "GPT2-with-RoPE"


# DDP 설정
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()  # 전체 DDP 프로세스 중 현재 프로세스의 번호(global rank) 0
world_size = dist.get_world_size()  # 전체 프로세스 수
is_ddp = world_size > 1

# 옵티마이저
OPTIMIZERS = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "sgd": optim.SGD,
}
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

# config 불러오기
_, value = parse_argument("--base-config")
base_config = load_yaml(value)

_, value = parse_argument("--experiment-config")
config = load_yaml(value)

_, value = parse_argument("--experiment")
experiment_config = config[value]

# seed 설정
seed = base_config["env"]["seed"]
set_seed(seed)

# 하이퍼파라미터 설정
vocab_size = base_config["model"]["vocab_size"]
seq_length = base_config["train"]["seq_length"]
d_model = base_config["train"]["d_model"]
d_ff = base_config["train"]["d_ff"]
lr = base_config["train"]["learning_rate"]
h = base_config["train"]["num_heads"]
num_layers = base_config["train"]["num_layers"]
grad_clip = base_config["train"]["grad_clip"]
weight_decay = base_config["train"]["weight_decay"]
micro_batch_size = experiment_config["micro_batch_size"]
precision = experiment_config["precision"]
dropout_rate = base_config["train"]["dropout_rate"]

# 기본 설정
_, value = parse_argument("--flag")
flag = value
_, value = parse_argument("--enable-compile")
enable_compile = value


# 실험 파라미터 설정
optimizer_steps = experiment_config["optimizer_steps"]
accumulation_steps = experiment_config["gradient_accumulation_steps"]

# Dataloader 생성
train_sampler, train_dataloader = get_dataloader(
    micro_batch_size, world_size, rank, split="train", shuffle=True, seed=seed
)

_, val_dataloader = get_dataloader(
    micro_batch_size, world_size, rank, split="val", shuffle=False, seed=seed
)

# 모델 생성
model = GPT2_RoPE(
    vocab_size=vocab_size,
    d_model=d_model,
    h=h,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=seq_length,
).to(device)
model.gradient_checkpointing = experiment_config["gradient_checkpointing"]
if is_ddp:
    model = DDP(model, device_ids=[local_rank])

if enable_compile:
    model = torch.compile(model)

# opmizer 생성
optimizer = OPTIMIZERS[base_config["train"]["optimizer"]](
    params=model.parameters(), lr=lr, weight_decay=weight_decay
)

# 손실함수 생성
loss_fn = nn.CrossEntropyLoss()

# 학습
train_losses = []
val_losses = []
eval_interval = base_config["train"]["eval_interval"]
micro_steps = optimizer_steps * accumulation_steps
model.train()
optimizer.zero_grad(set_to_none=True)

train_loss_sum = 0.0
optimizer_step = 0
epoch = 0
if train_sampler is not None:
    train_sampler.set_epoch(epoch)

train_iter = iter(train_dataloader)

if wandb_log and rank == 0:
    run = wandb.init(
        project=wandb_project,
        config={
            "model": wandb_run_name,
            "learning_rate": lr,
            "dropout_rate": dropout_rate,
            "architecture": "GPT-2 with RoPE",
            "dataset": "openwebtext",
            "compile": enable_compile,
        },
    )
    wandb_config = wandb.config
else:
    run = None

# 모니터링용 객체 생성
monitor = TrainingMonitor(run)

# T4에 FP16을 위한 스케일러(BF16에 대해서는 아무것도 안 함)
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(precision == "fp16"),
)
token_count = 0
monitor.start_throughput()
progress = (
    tqdm(range(1, (micro_steps + 1)), desc="진행률: ")
    if rank == 0
    else range(1, (micro_steps + 1))
)
for step in progress:
    try:
        x_batch, y_batch = next(train_iter)
    except StopIteration as e:
        # epoch이 끝났을 때 데이터를 균등하게 나누기 위해서 sampler를 다시 설정해야 함.
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)
        x_batch, y_batch = next(train_iter)

    update_now = step % accumulation_steps == 0
    # ddp인데 optimizer_step이 아니면 동기화하지 않음. optimizer_step이면 각 GPU에서 계산한 gradient를 통신해 평균내고, 모든 GPU가 같은 gradient를 갖도록 동기화하는 것
    sync_contex = model.no_sync() if is_ddp and not update_now else nullcontext()
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    with sync_contex:
        with autocast(device_type="cuda", dtype=DTYPE[precision]):

            logits = model(x_batch, mask=True)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            B, S, _ = logits.shape
            monitor.add_tokens(B * S)

            # 그래프가 누적되기 때문에 값만 누적하기 위해서 item을 사용
            train_loss_sum += loss.detach().item()
        scaler.scale(loss / accumulation_steps).backward()
        monitor.log_scale(
            rank=rank,
            scale=scaler.get_scale(),
            phase="train",
            stage="after backward",
            optimizer_step=optimizer_step,
            micro_step=step,
        )

    # AMP 학습에서는 FP32 파라미터와 상태로 업데이트하므로 autocast가 필요없음
    if update_now:
        # scaler에서 step함
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        monitor.log_throughput(
            rank=rank,
            phase="train",
            stage="after_optimizer_step",
            optimizer_step=optimizer_step,
            micro_step=step,
            flag=flag,
        )
        train_loss_tensor = torch.tensor(
            train_loss_sum, dtype=torch.float32, device=device
        )

        # tensor에 값을 적용
        if is_ddp:
            # all_reduce는 모든 GPU에 있는 텐서를 연산 후에 각 GPU에 동일한 텐서를 저장함
            dist.all_reduce(
                train_loss_tensor,
                op=torch.distributed.ReduceOp.SUM,
            )

        if rank == 0:
            avg_train_loss = (
                train_loss_tensor / (accumulation_steps * world_size)
            ).item()
            train_losses.append(avg_train_loss)
            monitor.log_loss(
                rank=rank,
                loss=avg_train_loss,
                phase="train",
                stage="calculate average train loss",
                optimizer_step=optimizer_step,
                micro_step=step,
                flag=flag,
            )
        train_loss_sum = 0.0

        if optimizer_step % eval_interval == 0:
            model.eval()
            val_loss_sum = 0.0
            batch_count = 0
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=DTYPE[precision]):
                    for x_batch, y_batch in val_dataloader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        logits = model(x_batch, mask=True)
                        loss = loss_fn(
                            logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1)
                        )
                        val_loss_sum += loss.item()
                        batch_count += 1

            val_loss_tensor = torch.tensor(
                val_loss_sum / batch_count, dtype=torch.float32, device=device
            )
            if is_ddp:
                dist.all_reduce(
                    val_loss_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                )
            if rank == 0:
                avg_val_loss = (val_loss_tensor / world_size).item()
                val_losses.append(avg_val_loss)
                parameters = {
                    "rank": rank,
                    "loss": avg_val_loss,
                    "phase": "validation",
                    "optimizer_step": optimizer_step,
                    "micro_step": step,
                    "flag": flag,
                }
                monitor.log_loss(
                    **parameters, stage="calculate average validation loss"
                )
                monitor.log_perplexity(**parameters, stage="calculate perplexity")
            model.train()
        monitor.start_throughput()
        if run is not None and rank:
            run.log({}, step=step, commit=True)


dist.destroy_process_group()
if run is not None and rank:
    run.finish()
