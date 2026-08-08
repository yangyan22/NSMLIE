import os
import time
import csv

class Logger:
    def __init__(self, save_dir, model_name, batch_size,  num_epochs, loss_weights,learning_rate,lr_decay,gamma, seed,enable_logging=True):
        self.enable_logging = enable_logging  # 是否启用日志记录

        if not self.enable_logging:
            return  # 如果关闭日志记录，则不初始化后续内容

        os.makedirs(save_dir, exist_ok=True)

        self.log_txt = os.path.join(save_dir, "log.txt")
        self.log_csv = os.path.join(save_dir, "log.csv")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_txt, "w") as f:
            f.write("=== Training Log ===\n")
            f.write(f"Time: {timestamp}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Total epochs: {num_epochs}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"lr_decay: {lr_decay}\n")
            f.write(f"gamma: {gamma}\n")
            f.write(f"seed:{seed}\n")
            f.write("Loss Weights:\n")
            for name, value in loss_weights.items():
                f.write(f"  {name}: {value}\n")
            f.write("=" * 50 + "\n\n")


        self.loss_keys = list(loss_weights.keys())
        self.csv_header = ["epoch", "iteration", "total_loss", *self.loss_keys, "lr"]

        with open(self.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)

    def log(self, epoch, iteration, total_iter, total_loss, loss_dict, lr):
        if not self.enable_logging:
            return  # 如果关闭日志，不执行任何写入

        # 写文本日志
        loss_values_str = " | ".join(
            f"{k}: {loss_dict.get(k, 0):.4f}" for k in self.loss_keys
        )
        log_line = (
            f"===> Epoch[{epoch:03d}]({iteration:03d}/{total_iter:03d}) "
            f"Total Loss: {total_loss:.4f} || {loss_values_str} || LR: {lr:.6f}\n"
        )
        with open(self.log_txt, "a") as f:
            f.write(log_line)


