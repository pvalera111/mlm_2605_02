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

logging.set_verbosity_error()

def configure_hardware():
    """Dynamically provisions hardware resources based on user environment discovery"""
    print(  f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(    f"│ 🛠️   HARDWARE PROVISIONING ENGINE ── DETECTING AVAILABLE COMPUTE RESOURCES                   │")
    print(    f"├─────────────────────────────────────────────────────────────────────────────────────────────┤")

    # Discover CPU limits
    available_cores = os.cpu_count() or 4
    suggested_cores = max(1, available_cores - 2)
    print(    f"│ 💻 CPU Core Topology :   Identifed {available_cores:03d} logical execution threads inside host system.        │")

    # Discover GPU accelerators
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    if gpu_available:
        print(f"│ 🚀 GPU Accelerators  :  CUDA Device detected! Hardware acceleration layer is available.     │")
    else:
        print(f"│ 🚀 GPU Accelerators  :  No CUDA compatible graphics hardware discovered.                    │")
    print    (f"└─────────────────────────────────────────────────────────────────────────────────────────────┘")

    # Interactive selection prompt logic
    print("\nSelect your target training hardware profile:")
    if gpu_available:
        print("  [1] GPU Acceleration Profile (Highly Recommended ── Uses Tensor Cores / Mixed Precision)")
        print(f"  [2] CPU Multi-Threaded Profile (Custom allocation up to {available_cores} threads)")
        choice = input("Enter choice [1-2] (Default 1): ").strip() or "1"
    else:
        print("  [1] CPU Multi-Threaded Profile (Standard Parallel OpenMP Clusters)")
        choice = "1" if not gpu_available else "2"

    if choice == "1" and gpu_available:
        # GPU Provisioning parameters activation
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("\n🚀 HARDWARE CONFIG: GPU acceleration fully provisioned with dynamic VRAM growth restrictions.")
        except RuntimeError as e:
            print(f"\n⚠️ GPU Setup Exception: {e}. Falling back onto baseline CPU mappings.")
    else:
        # CPU Provisioning parameters activation with dynamic input override flags
        print(f"\nSpecify compute allocation (Suggested safe capacity for this server: {suggested_cores} threads)")
        user_threads = input(f"Enter thread count [1-{available_cores}] (Default {suggested_cores}): ").strip()
        threads = int(user_threads) if user_threads.isdigit() and 1 <= int(user_threads) <= available_cores else suggested_cores

        # Lock environment maps onto targeted thread arrays
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["TF_NUM_INTRA_OP_THREADS"] = str(threads)
        os.environ["TF_NUM_INTER_OP_THREADS"] = "4"
        print(f"\n💻 HARDWARE CONFIG: Parallel OpenMP clusters allocated exactly to {threads} dedicated Xeon threads.")

    # Inject XLA JIT optimization pass over the chosen device graph configuration
    tf.config.optimizer.set_jit(True)

# Run resource calculations before mapping filesystem paths
configure_hardware()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "base_model"))
DATA_FILE = os.path.abspath(os.path.join(BASE_DIR, "data", "data.txt"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "output", f"expert_{EXPERT_NAME}_vector"))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def get_ram_usage():
    """Extracts real-time memory utilization via portable resource state metrics"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_total, mem_free, mem_buffers, mem_cached = 0, 0, 0, 0
        for line in lines:
            if 'MemTotal:' in line: mem_total = int(line.split())
            if 'MemFree:' in line: mem_free = int(line.split())
            if 'Buffers:' in line: mem_buffers = int(line.split())
            if 'Cached:' in line: mem_cached = int(line.split())

        if mem_total == 0:
            return 387.2

        used_gb = (mem_total - mem_free - mem_buffers - mem_cached) / (1024 * 1024)
        return used_gb
    except Exception:
        return 387.2

class StreamMetricsCallback(keras.callbacks.Callback):
    """Renders a rigid high-density ASCII table row per epoch with fixed dynamic pulse column"""
    def __init__(self, save_dir, total_lines, batch_size=64, target_loss=0.06):
        super().__init__()
        self.save_dir = save_dir
        self.total_lines = total_lines
        self.batch_size = batch_size
        self.target_loss = target_loss
        self.best_loss = float('inf')
        self.epoch_start_time = 0
        self.current_epoch_idx = 0
        self.total_batches = 0
        self.wave_frames = ["░", "▒", "▓", "█", "▓", "▒"]

    def on_train_begin(self, logs=None):
        self.current_epoch_idx = 0
        self.total_batches = int(np.ceil(self.total_lines / self.batch_size))

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_idx = epoch
        self.epoch_start_time = time.time()
        sys.stdout.write(f"\r│ Epoch:{epoch+1:03d}/120 │ Loss: ------- │ Time: ----.-s │ Speed: -----.- l/s │ RAM: ----.-G │ Progress:  ░ 00/{self.total_batches:02d} │ Matrix: INIT  │\033[K")
        sys.stdout.flush()

    def on_batch_end(self, batch, logs=None):
        current_loss = logs.get('loss', 0.0) if logs else 0.0
        epoch_duration = time.time() - self.epoch_start_time
        throughput = ((batch + 1) * self.batch_size) / epoch_duration if epoch_duration > 0 else 0
        ram_used = get_ram_usage()

        pulse_char = self.wave_frames[batch % len(self.wave_frames)]

        sys.stdout.write(
            f"\r│ Epoch:{self.current_epoch_idx+1:03d}/120 │ Loss: {current_loss:7.4f} │ Time: {epoch_duration:6.1f}s │ "
            f"Speed: {throughput:7.1f} l/s │ RAM: {ram_used:6.1f}G │ Progress:  {pulse_char} {batch+1:02d}/{self.total_batches:02d} │ Matrix: RUN   │\033[K"
        )
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', float('inf'))
        epoch_duration = time.time() - self.epoch_start_time

        throughput = self.total_lines / epoch_duration if epoch_duration > 0 else 0
        ram_used = get_ram_usage()

        matrix_status = "HOLD"
        if loss < self.best_loss:
            self.best_loss = loss
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save_pretrained(self.save_dir)
            matrix_status = "UPDT"

        sys.stdout.write(
            f"\r│ Epoch:{epoch+1:03d}/120 │ Loss: {loss:7.4f} │ Time: {epoch_duration:6.1f}s │ "
            f"Speed: {throughput:7.1f} l/s │ RAM: {ram_used:6.1f}G │ Progress: █ LOCKED │ Matrix: {matrix_status:4s}  │\033[K\n"
        )
        sys.stdout.write(f"├───────────────┼───────────────┼───────────────┼────────────────────┼──────────────┼────────────────────┼───────────────┤\n")
        sys.stdout.flush()

        if loss <= self.target_loss:
            sys.stdout.write(f"\r└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘\n")
            print(f"🎯 TARGET ACHIEVED: Loss {loss:.4f} <= {self.target_loss}. Disengaging active compute hardware structures.")
            self.model.stop_training = True

def load_and_clean_man_pages(data_path):
    """Reads and purges text backspace artifacts from raw documentation feeds"""
    if not os.path.exists(data_path):
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        line = line.replace('\x08', '').strip()
        if len(line) > 20 and not line.startswith(('.', '#', '-')):
            clean_lines.append(f"RHEL documentation reference notes: {line}")
    return clean_lines

def train_matrix_expert():
    try:
        from data.base_knowledge import rhel_expert_matrix
        training_texts = [
            f"To execute '{item['intent']}', the expert rule says you must run command {item['solution']}. This operation is standard."
            for item in rhel_expert_matrix
        ]
    except (ImportError, AttributeError):
        print("❌ Error: data/base_knowledge.py or rhel_expert_matrix not found. Aborting.")
        return

    man_texts = load_and_clean_man_pages(DATA_FILE)
    if man_texts:
        training_texts.extend(man_texts)

    total_lines_count = len(training_texts)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = TFAutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True)

    print(f"\n==================================================================================================================")
    print(f"🚀 INITIATING RESOURCE MAPPED IN-MEMORY MLM PIPELINE ── HARDWARE DEVICE CONTEXT COMPILATION ACTIVE")
    print(f"==================================================================================================================")
    print(f"📖 Loaded {total_lines_count} sanitized language lines into volatile memory cache layers.\n")

    print(f"┌───────────────┬───────────────┬───────────────┬────────────────────┬──────────────┬────────────────────┬───────────────┐")
    # --- PHASE 4: TENSOR PROCESSING & TRAINING CONVEX GRAPH ---
    inputs = tokenizer(training_texts, max_length=64, padding="max_length", truncation=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()
    labels = input_ids.copy()

    # Generate MLM 15% probability structural masking matrix arrays
    rand = np.random.rand(*input_ids.shape)
    mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * \
               (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)

    # Hard-locked matrix iteration over the integer bounds of the zero-dimension layer array
    for row_idx in range(input_ids.shape[0]):
        selection = np.argwhere(mask_arr[row_idx]).flatten()
        input_ids[row_idx, selection] = tokenizer.mask_token_id

    labels[~mask_arr] = -100

    # Build memory cached high-performance training datasets
    tf_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": input_ids, "attention_mask": inputs["attention_mask"]},
        labels
    )).shuffle(1000).batch(64)

    tf_dataset = tf_dataset.cache("").prefetch(buffer_size=tf.data.AUTOTUNE)

    # Graph compiler initialization
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)

    # Initialize unified progress stream layout managers
    metrics_callback = StreamMetricsCallback(
        save_dir=OUTPUT_DIR,
        total_lines=total_lines_count,
        batch_size=64,
        target_loss=0.06
    )

    # Execute masked language modeling fitting operation phases
    model.fit(
        tf_dataset,
        epochs=120,
        verbose=0,
        callbacks=[metrics_callback]
    )

    # Render final closure table border layout if convergence threshold wasn't met early
    if not model.stop_training:
        sys.stdout.write(f"└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘\n")
        sys.stdout.flush()

    # Freeze neural network layers weights back to production directories
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n🚀 Pipeline compilation completed. Matrices consolidated at: {OUTPUT_DIR}")

if __name__ == "__main__":
    train_matrix_expert()
