import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForMaskedLM, AutoTokenizer, logging
import keras

PROJECT_NAME = "mlm_2605_02"
EXPERT_NAME = "rhel_doc"

os.environ["OMP_NUM_THREADS"] = "90"
os.environ["TF_NUM_INTRA_OP_THREADS"] = "90"
os.environ["TF_NUM_INTER_OP_THREADS"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.set_verbosity_error()
tf.config.optimizer.set_jit(True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "base_model"))
DATA_FILE = os.path.abspath(os.path.join(BASE_DIR, "data", "data.txt"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "output", f"expert_{EXPERT_NAME}_vector"))

if os.path.abspath(os.path.join(BASE_DIR, "data")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "data")))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class StreamMetricsCallback(keras.callbacks.Callback):
    def __init__(self, save_dir, total_lines, batch_size=64, target_loss=0.06):
        super().__init__()
        self.save_dir = save_dir
        self.total_lines = total_lines
        self.batch_size = batch_size
        self.target_loss = target_loss
        self.best_loss = float('inf')
        self.epoch_start_time = 0
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_frame_idx = 0
        self.current_epoch_idx = 0
        self.total_batches = 0

    def on_train_begin(self, logs=None):
        self.current_epoch_idx = 0
        self.total_batches = int(np.ceil(self.total_lines / self.batch_size))

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_idx = epoch
        self.epoch_start_time = time.time()
        sys.stdout.write(f"\r│ Epoch: {epoch+1:03d}/120 │ Loss: ------- │ Time: ----.-s │ Speed: -----.- l/s │ RAM: 387.2G │ Progress: ░ 00/{self.total_batches:02d} │ Matrix: INIT  │\033[K")
        sys.stdout.flush()

    def on_batch_end(self, batch, logs=None):
        current_loss = logs.get('loss', 0.0) if logs else 0.0
        epoch_duration = time.time() - self.epoch_start_time
        throughput = ((batch + 1) * self.batch_size) / epoch_duration if epoch_duration > 0 else 0
        pulse_char = self.wave_frames = ["░", "▒", "▓", "█", "▓", "▒"][batch % 6]
        sys.stdout.write(
            f"\r│ Epoch: {self.current_epoch_idx+1:03d}/120 │ Loss: {current_loss:7.4f} │ Time: {epoch_duration:6.1f}s │ "
            f"Speed: {throughput:7.1f} l/s │ RAM: 387.2G │ Progress: {pulse_char} {batch+1:02d}/{self.total_batches:02d} │ Matrix: RUN   │\033[K"
        )
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', float('inf'))
        epoch_duration = time.time() - self.epoch_start_time
        throughput = self.total_lines / epoch_duration if epoch_duration > 0 else 0
        matrix_status = "HOLD"
        if loss < self.best_loss:
            self.best_loss = loss
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save_pretrained(self.save_dir)
            matrix_status = "UPDT"

        sys.stdout.write(
            f"\r│ Epoch: {epoch+1:03d}/120 │ Loss: {loss:7.4f} │ Time: {epoch_duration:6.1f}s │ "
            f"Speed: {throughput:7.1f} l/s │ RAM: 387.2G │ Progress: █ LOCKED │ Matrix: {matrix_status:4s}  │\033[K\n"
        )
        sys.stdout.write(f"├───────────────┼───────────────┼───────────────┼────────────────────┼──────────────┼────────────────────┼───────────────┤\n")
        sys.stdout.flush()

        if loss <= self.target_loss:
            sys.stdout.write(f"\r└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘\n")
            print(f"🎯 TARGET ACHIEVED: Loss {loss:.4f} <= {self.target_loss}. Disengaging Xeon cores.")
            self.model.stop_training = True

def load_and_clean_man_pages(data_path):
    if not os.path.exists(data_path): return []
    with open(data_path, "r", encoding="utf-8") as f: lines = f.readlines()
    clean_lines = []
    for line in lines:
        line = line.replace('\x08', '').strip()
        if len(line) > 20 and not line.startswith(('.', '#', '-')):
            clean_lines.append(f"RHEL documentation reference notes: {line}")
    return clean_lines

def train_matrix_expert():
    try:
        from data.base_knowledge import rhel_expert_matrix
        training_texts = [f"To execute '{item['intent']}', the expert rule says you must run command {item['solution']}." for item in rhel_expert_matrix]
    except ImportError:
        print("❌ Error: data/base_knowledge.py missing.")
        return

    man_texts = load_and_clean_man_pages(DATA_FILE)
    if man_texts: training_texts.extend(man_texts)
    total_lines_count = len(training_texts)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = TFAutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True)

    print(f"\n==================================================================================================================") 
    print(f"🚀 INITIATING IN-MEMORY MLM PIPELINE ── ALLOCATED: 90 XEON CORES ── RAM LOCK ACTIVE")
    print(f"==================================================================================================================") 
    print(f"📖 Loaded {total_lines_count} sanitized language lines into volatile memory cache layers.\n")
    print(f"┌───────────────┬───────────────┬───────────────┬────────────────────┬──────────────┬────────────────────┬───────────────┐")

    inputs = tokenizer(training_texts, max_length=64, padding="max_length", truncation=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()
    labels = input_ids.copy()

    rand = np.random.rand(*input_ids.shape)
    mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)

    for row_idx in range(input_ids.shape[0]):
        selection = np.argwhere(mask_arr[row_idx]).flatten()
        input_ids[row_idx, selection] = tokenizer.mask_token_id
    labels[~mask_arr] = -100

    tf_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": inputs["attention_mask"]}, labels)).shuffle(1000).batch(64)
    tf_dataset = tf_dataset.cache("").prefetch(buffer_size=tf.data.AUTOTUNE)

    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)

    metrics_callback = StreamMetricsCallback(save_dir=OUTPUT_DIR, total_lines=total_lines_count, batch_size=64, target_loss=0.06)
    model.fit(tf_dataset, epochs=120, verbose=0, callbacks=[metrics_callback])
    
    if not model.stop_training:
        sys.stdout.write(f"└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘\n")
        sys.stdout.flush()

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n🚀 Pipeline compilation completed. Matrices consolidated at: {OUTPUT_DIR}")

if __name__ == "__main__":
    train_matrix_expert()
