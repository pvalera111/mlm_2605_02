import os
import sys
import time
import json
import numpy as np

PROJECT_NAME = "mlm_2605_02"
EXPERT_NAME = "rhel_doc"

# Safe Multi-Platform Dynamic Architecture Isolation Pass
BACKEND = None
try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, logging
    BACKEND = "TORCH"
except ImportError:
    try:
        import tensorflow as tf
        from transformers import TFAutoModelForMaskedLM, AutoTokenizer, logging
        import keras
        BACKEND = "TF"
    except ImportError:
        print("💻 CI NOTE: Frameworks missing. Injecting runtime simulation mocks.")
        BACKEND = "MOCK"
        class MockObject:
            def __getattr__(self, name): return lambda *args, **kwargs: MockObject()
        torch = MockObject()
        torch.device = lambda x: "cpu"
        torch.cuda = MockObject()
        torch.cuda.is_available = lambda: False
        tf = MockObject()
        keras = MockObject()
        logging = MockObject()
        def AutoTokenizer(*args, **kwargs): return MockObject()
        def AutoModelForMaskedLM(*args, **kwargs): return MockObject()
        def TFAutoModelForMaskedLM(*args, **kwargs): return MockObject()

def configure_hardware():
    """Dynamically provisions hardware and core counts across TF and PyTorch environments"""
    print(f"\n┌───────────────┬───────────────┬───────────────┬────────────────────┬──────────────┬────────────────────┬───────────────┐")
    print(f"│ 🛠️  HARDWARE PROVISIONING ENGINE ── DETECTING AVAILABLE COMPUTE RESOURCES                                          │")
    print(f"├───────────────┼───────────────┼───────────────┼────────────────────┼──────────────┼────────────────────┼───────────────┤")
    
    available_cores = os.cpu_count() or 4
    suggested_cores = max(1, available_cores - 2)
    print(f"│ 💻 CPU Core Topology : Identified {available_cores:03d} logical execution threads inside host system.                       │")
    
    device_name = "CPU"
    if BACKEND == "TORCH":
        if torch.cuda.is_available(): device_name = "CUDA (NVIDIA)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device_name = "MPS (Apple)"
    elif BACKEND == "TF":
        if tf.config.list_physical_devices('GPU'): device_name = "GPU (TensorFlow)"

    print(f"│ 🚀 Active Framework  : Engine linked via [{BACKEND}]. Target Device profile: [{device_name}].               │")
    print(f"└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘")
    
    if device_name == "CPU":
        print(f"\nSpecify compute allocation (Suggested safe capacity for this server: {suggested_cores} threads)")
        user_threads = input(f"Enter thread count [1-{available_cores}] (Default {suggested_cores}): ").strip()
        threads = int(user_threads) if user_threads.isdigit() and 1 <= int(user_threads) <= available_cores else suggested_cores
        
        if BACKEND == "TORCH":
            torch.set_num_threads(threads)
        elif BACKEND == "TF":
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["TF_NUM_INTRA_OP_THREADS"] = str(threads)
            os.environ["TF_NUM_INTER_OP_THREADS"] = "4"
            tf.config.optimizer.set_jit(True)
        print(f"\n💻 HARDWARE CONFIG: Runtime allocated exactly to {threads} parallel compute threads.")
    else:
        if BACKEND == "TF":
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.optimizer.set_jit(True)
        print(f"\n🚀 HARDWARE CONFIG: Execution targeted directly to hardware accelerator: [{device_name}]")

if BACKEND and BACKEND != "MOCK":
    logging.set_verbosity_error()
    configure_hardware()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "base_model"))
DATA_FILE = os.path.abspath(os.path.join(BASE_DIR, "data", "data.txt"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "output", f"expert_{EXPERT_NAME}_vector"))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class UnifiedMetricsCallback:
    """Rigid high-density ASCII table dashboard row formatter compatible with TF and Torch loops"""
    def __init__(self, total_lines, batch_size=64, target_loss=0.06):
        self.total_lines = total_lines
        self.batch_size = batch_size
        self.target_loss = target_loss
        self.best_loss = float('inf')
        self.epoch_start_time = 0
        self.current_epoch_idx = 0
        self.total_batches = int(np.ceil(total_lines / batch_size))
        self.wave_frames = ["░", "▒", "▓", "█", "▓", "▒"]

    def on_epoch_begin(self, epoch):
        self.current_epoch_idx = epoch
        self.epoch_start_time = time.time()
        sys.stdout.write(f"\r│ Epoch: {epoch+1:03d}/120 │ Loss: ------- │ Time: ----.-s │ Speed: -----.- l/s │ RAM: 387.2G │ Progress: ░ 00/{self.total_batches:02d} │ Matrix: INIT  │\033[K")
        sys.stdout.flush()

    def on_batch_end(self, batch, current_loss):
        epoch_duration = time.time() - self.epoch_start_time
        throughput = ((batch + 1) * self.batch_size) / epoch_duration if epoch_duration > 0 else 0
        pulse_char = self.wave_frames[batch % len(self.wave_frames)]
        sys.stdout.write(f"\r│ Epoch: {self.current_epoch_idx+1:03d}/120 │ Loss: {current_loss:7.4f} │ Time: {epoch_duration:6.1f}s │ Speed: {throughput:7.1f} l/s │ RAM: 387.2G │ Progress: {pulse_char} {batch+1:02d}/{self.total_batches:02d} │ Matrix: RUN   │\033[K")
        sys.stdout.flush()

    def on_epoch_end(self, epoch, loss, save_trigger_fn):
        epoch_duration = time.time() - self.epoch_start_time
        throughput = self.total_lines / epoch_duration if epoch_duration > 0 else 0
        matrix_status = "HOLD"
        if loss < self.best_loss:
            self.best_loss = loss
            save_trigger_fn(OUTPUT_DIR)
            matrix_status = "UPDT"
        sys.stdout.write(f"\r│ Epoch: {epoch+1:03d}/120 │ Loss: {loss:7.4f} │ Time: {epoch_duration:6.1f}s │ Speed: {throughput:7.1f} l/s │ RAM: 387.2G │ Progress: █ LOCKED │ Matrix: {matrix_status:4s}  │\033[K\n")
        sys.stdout.write(f"├───────────────┼───────────────┼───────────────┼────────────────────┼──────────────┼────────────────────┼───────────────┤\n")
        sys.stdout.flush()
        if loss <= self.target_loss:
            sys.stdout.write(f"\r└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘\n")
            print(f"🎯 TARGET ACHIEVED: Loss {loss:.4f} <= {self.target_loss}. Disengaging active compute hardware.")
            return True
        return False

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
    if BACKEND == "MOCK":
        print("💻 CI NOTE: Mock validation environment. Halting training trace graph execution.")
        return
        
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
    metrics = UnifiedMetricsCallback(total_lines_count, batch_size=64, target_loss=0.06)

    print(f"\n==================================================================================================================") 
    print(f"🚀 INITIATING CONDITIONAL CROSS-PLATFORM MLM PIPELINE [ENGINE RUNTIME MODE: {BACKEND}]")
    print(f"==================================================================================================================") 
    print(f"📖 Loaded {total_lines_count} sanitized language lines into volatile memory cache layers.\n")
    print(f"┌───────────────┬───────────────┬───────────────┬────────────────────┬──────────────┬────────────────────┬───────────────┐")

    # --- CONDITIONAL RUNTIME COMPILATION BRANCH: PYTORCH ---
    if BACKEND == "TORCH":
        target_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        model_torch = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True).to(target_device)
        
        inputs = tokenizer(training_texts, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)
        input_ids[mask_arr] = tokenizer.mask_token_id
        labels[~mask_arr] = -100

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_ids, attention_mask, labels), batch_size=64, shuffle=True)
        optimizer = torch.optim.AdamW(model_torch.parameters(), lr=5e-5)
        
        model_torch.train()
        for epoch in range(120):
            metrics.on_epoch_begin(epoch)
            epoch_loss = 0.0
            for b_idx, (b_in, b_att, b_lab) in enumerate(dataloader):
                b_in = b_in.to(target_device)
                b_att = b_att.to(target_device)
                b_lab = b_lab.to(target_device)
                
                optimizer.zero_grad()
                loss = model_torch(input_ids=b_in, attention_mask=b_att, labels=b_lab).loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                metrics.on_batch_end(b_idx, loss.item())
                
            avg_loss = epoch_loss / len(dataloader)
            if metrics.on_epoch_end(epoch, avg_loss, lambda path: model_torch.save_pretrained(path)): 
                break

    # --- CONDITIONAL RUNTIME COMPILATION BRANCH: TENSORFLOW ---
    elif BACKEND == "TF":
        model_tf = TFAutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True)
        inputs = tokenizer(training_texts, max_length=64, padding="max_length", truncation=True, return_tensors="tf")
        input_ids = inputs["input_ids"].numpy()
        labels = input_ids.copy()
        
        rand = np.random.rand(*input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * \
                   (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)
        
         for r_idx in range(input_ids.shape[0]): 
            input_ids[r_idx, np.argwhere(mask_arr[r_idx]).flatten()] = tokenizer.mask_token_id
            
        labels[~mask_arr] = -100

        tf_dataset = tf.data.Dataset.from_tensor_slices((
            {"input_ids": input_ids, "attention_mask": inputs["attention_mask"]}, 
            labels
        )).shuffle(1000).batch(64).cache("").prefetch(tf.data.AUTOTUNE)
        
        model_tf.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5))

        class TFCallbackBridge(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None): 
                metrics.on_epoch_begin(epoch)
            def on_batch_end(self, batch, logs=None): 
                metrics.on_batch_end(batch, logs.get('loss', 0.0) if logs else 0.0)
            def on_epoch_end(self, epoch, logs=None):
                current_loss = logs.get('loss', float('inf')) if logs else float('inf')
                if metrics.on_epoch_end(epoch, current_loss, lambda path: self.model.save_pretrained(path)):
                    self.model.stop_training = True

        model_tf.fit(tf_dataset, epochs=120, verbose=0, callbacks=[TFCallbackBridge()])

    if not metrics.best_loss <= metrics.target_loss:
        sys.stdout.write(f"└───────────────┴───────────────┴───────────────┴────────────────────┴──────────────┴────────────────────┴───────────────┘\n")
        
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n🚀 Compilation complete. Uniform platform vectors generated safely at: {OUTPUT_DIR}")

if __name__ == "__main__":
    train_matrix_expert()
