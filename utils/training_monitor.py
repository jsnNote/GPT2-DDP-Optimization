import torch
import torch.distributed as dist
from datetime import datetime
from collections import defaultdict
import json
import time
from tqdm import tqdm
import os

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)


class ConsoleFormatter:
    def format(self, record):
        unit = record["unit"] if record["unit"] is not None else ""
        string = str()
        for name, value in record["value"].items():
            string += f" | {name}={value:.2f}{unit}"

        return (
            f"[{record['level']}] {record['time']} | "
            f"metric={record['metric']} | "
            f"phase={record['phase']} | "
            f"stage={record['stage']} | "
            f"device={record['device']} | rank={record['rank']} | "
            f"optimizer_step={record['optimizer_step']} | "
            f"micro_step={record['micro_step']}"
            f"{string}"
        )


class TrainingMonitor:
    def __init__(self, run=None, is_ddp=False):
        self.log = defaultdict(list)
        self._throughput_start_time = None
        self._console_format = ConsoleFormatter()
        self._pending_logs = []
        self._local_tokens = 0
        self._run = run
        self._is_ddp = is_ddp

    def __get_time(self):
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def log_gpu_memory(
        self,
        rank,
        phase=None,
        stage=None,
        optimizer_step=None,
        micro_step=None,
        flag=False,
    ):
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        mib = 1024**2
        world_size = dist.get_world_size()
        local_utilization = torch.cuda.utilization(device) / mib
        local_allocated = torch.cuda.memory_allocated(device) / mib
        local_reserved = torch.cuda.memory_reserved(device) / mib
        local_peak_allocated = torch.cuda.max_memory_allocated(device) / mib

        local_metrics = torch.tensor(
            [
                local_utilization,
                local_allocated,
                local_reserved,
                local_peak_allocated,
            ],
            dtype=torch.float64,
            device=device,
        )

        gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(world_size)]
        if self._is_ddp:
            dist.all_gather(gathered_metrics, local_metrics)
        else:
            gathered_metrics = [local_metrics]

        record = {
            "level": "INFO",
            "metric": "gpu_memory",
            "time": self.__get_time(),
            "phase": phase,
            "stage": stage,
            "rank": rank,
            "device": device,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "unit": "MiB",
            "value": {
                "utilization": local_utilization,
                "allocated": local_allocated,
                "reserved": local_reserved,
                "peak_allocated": local_peak_allocated,
            },
        }
        self.log["gpu_memory"].append(record)
        self._pending_logs.append(record)

        if rank == 0:
            metrics = torch.stack(gathered_metrics)

            # 열별 집계
            global_utilization_mean = metrics[:, 0].mean().item()
            global_allocated_max = metrics[:, 1].max().item()
            global_reserved_max = metrics[:, 2].max().item()
            global_peak_max = metrics[:, 3].max().item()

            data = {
                "gpu/utilization/mean": global_utilization_mean,
                "gpu/memory_allocated/max": global_allocated_max,
                "gpu/memory_reserved/max": global_reserved_max,
                "gpu/memory_peak/max": global_peak_max,
            }

            # Rank별 기록
            for r in range(world_size):
                data[f"gpu/utilization/rank_{r}"] = metrics[r, 0].item()
                data[f"gpu/memory_allocated/rank_{r}"] = metrics[r, 1].item()
                data[f"gpu/memory_reserved/rank_{r}"] = metrics[r, 2].item()
                data[f"gpu/memory_peak/rank_{r}"] = metrics[r, 3].item()

            self._run.log(data=data, step=micro_step)

        if flag and rank == 0:
            tqdm.write(self._console_format.format(record))

    def log_loss(
        self,
        rank,
        loss,
        phase=None,
        stage=None,
        optimizer_step=None,
        micro_step=None,
        flag=False,
    ):
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        record = {
            "level": "INFO",
            "metric": f"{phase}_loss",
            "time": self.__get_time(),
            "phase": phase,
            "stage": stage,
            "rank": rank,
            "device": device,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "unit": None,
            "value": {"loss": loss},
        }
        self.log[f"{phase}_loss"].append(record)
        self._pending_logs.append(record)

        if self._run is not None and rank == 0:
            data = {}
            for name, value in record["value"].items():
                data[f"{phase}/{name}/global"] = value
            self._run.log(data=data, step=micro_step, commit=False)

        if flag and rank == 0:
            tqdm.write(self._console_format.format(record))

    def start_throughput(
        self,
    ):
        torch.cuda.synchronize()
        self._throughput_start_time = time.perf_counter()

    def add_tokens(self, count):
        self._local_tokens += count

    def log_throughput(
        self,
        rank,
        phase=None,
        stage=None,
        optimizer_step=None,
        micro_step=None,
        flag=False,
    ):
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        torch.cuda.synchronize()
        local_duration = time.perf_counter() - self._throughput_start_time
        local_TPS = self._local_tokens / local_duration

        # global TPS 측정을 위한 코드
        tokens_tensor = torch.tensor(
            self._local_tokens, dtype=torch.float64, device=device
        )
        duration_tensor = torch.tensor(
            local_duration, dtype=torch.float64, device=device
        )

        if self._is_ddp:
            dist.all_reduce(tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(duration_tensor, op=torch.distributed.ReduceOp.MAX)

            # local rank의 TPS 모으기
            local_TPS_tensor = torch.tensor(device=device)
            gathered_tps = dist.all_gather(local_TPS_tensor, local_TPS)
        else:
            gathered_tps = torch.tensor([local_TPS], device=device)
        record = {
            "level": "INFO",
            "metric": "throughput",
            "time": self.__get_time(),
            "phase": phase,
            "stage": stage,
            "rank": rank,
            "device": device,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "unit": "token/sec",
            "value": {
                "TPS": local_TPS,
            },
        }
        self.log[f"throughput"].append(record)
        self._pending_logs.append(record)

        if self._run is not None and rank == 0:
            data = {}
            # local TPS
            for r, tps in enumerate(gathered_tps):
                data[f"{phase}/TPS/rank_{r}"] = tps.item()

            # global TPS
            if self._is_ddp:
                data[f"{phase}/TPS/global"] = (tokens_tensor / duration_tensor).item()
            self._run.log(data=data, step=micro_step)

        if flag and rank == 0:
            tqdm.write(self._console_format.format(record))

        self._local_tokens = 0
        self._throughput_start_time = None

    def log_perplexity(
        self,
        rank,
        loss,
        phase=None,
        stage=None,
        optimizer_step=None,
        micro_step=None,
        flag=False,
    ):
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        torch.cuda.synchronize()
        PPL = torch.exp(torch.tensor(loss))
        record = {
            "level": "INFO",
            "metric": "perplexity",
            "time": self.__get_time(),
            "phase": phase,
            "stage": stage,
            "rank": rank,
            "device": device,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "unit": None,
            "value": {
                "perplexity": PPL.item(),
            },
        }
        self.log[f"perplexity"].append(record)
        self._pending_logs.append(record)

        if self._run is not None and rank == 0:
            data = {}
            for name, value in record["value"].items():
                data[f"{phase}/{name}/global"] = value
            self._run.log(data=data, step=micro_step)

        if flag and rank == 0:
            tqdm.write(self._console_format.format(record))

    def save_log(self, path="log.jsonl"):

        with open(path, "a", encoding="utf-8") as f:
            for log in self._pending_logs:
                line = json.dumps(
                    log,
                    ensure_ascii=False,
                )
                f.write(line + "\n")
            f.flush()
            self._pending_logs.clear()
        print(f"[TrainingMonitor] {path}에 저장 완료")
