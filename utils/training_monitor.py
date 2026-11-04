import torch
import torch.distributed as dist
from datetime import datetime
from collections import defaultdict
import json
import time
from tqdm import tqdm


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
    def __init__(self):
        self.log = defaultdict(list)
        self._throughput_start_time = None
        self._console_format = ConsoleFormatter()
        self._pending_logs = []

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
                # "allocated": torch.cuda.memory_allocated(device) / mib,
                # "reserved": torch.cuda.memory_reserved(device) / mib,
                # "peak_allocated": torch.cuda.max_memory_allocated(device) / mib,
                "allocated": 10 / mib,
                "reserved": 20 / mib,
                "peak_allocated": 30 / mib,
            },
        }
        self.log["gpu_memory"].append(record)
        self._pending_logs.append(record)
        if flag and dist.get_rank() == 0:
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
            "value": {
                "loss": loss.item(),
            },
        }
        self.log[f"{phase}_loss"].append(record)
        self._pending_logs.append(record)

        if flag and dist.get_rank() == 0:
            tqdm.write(self._console_format.format(record))

    def start_throughput(
        self,
    ):
        self._throughput_start_time = time.perf_counter()

    def log_throughput(
        self,
        rank,
        token_count,
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
        # torch.cuda.synchronize()
        duration = time.perf_counter() - self._throughput_start_time
        TPS = token_count / duration
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
                "token_per_sec": TPS,
            },
        }
        self.log[f"throughput"].append(record)
        self._pending_logs.append(record)

        if flag and dist.get_rank() == 0:
            tqdm.write(self._console_format.format(record))

        self._throughput_start_time = None

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


monitor = TrainingMonitor()

monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
monitor.start_throughput()
monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
import time

time.sleep(10)
monitor.log_throughput(
    0,
    1000,
    "train",
    "after_optimizer_step",
    optimizer_step=4,
    micro_step=128,
    flag=True,
)
monitor.log_gpu_memory(0, "train", "after_forward", 4, 128, flag=True)
monitor.log_loss(0, torch.tensor(81.5), "train", "after_forward", 4, 128, flag=True)
monitor.save_log()
