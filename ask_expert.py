import os
import sys
import json
import numpy as np

# Adaptive backend framework parsing routing hooks
BACKEND = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    BACKEND = "TORCH"
except ImportError:
    try:
        import tensorflow as tf
        from transformers import AutoTokenizer, TFAutoModelForMaskedLM
        BACKEND = "TF"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except ImportError:
        pass

import readline
readline.parse_and_bind("tab: complete")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
MODEL_PATH = os.path.join(BASE_DIR, "output", "expert_rhel_doc_vector")
RULES_FILE = os.path.join(BASE_DIR, "data", "routing_rules.json")

def get_mean_pooling_vector(text, tokenizer, embedding_model):
    stop_words = {"what", "command", "can", "i", "write", "to", "for", "how", "show", "get", "view", "see", "print", "display"}
    words = [w for w in text.lower().split() if w not in stop_words]
    clean_text = " ".join(words) if words else text

    # Execution path for PyTorch structures
    if BACKEND == "TORCH":
        inputs = tokenizer(clean_text, max_length=45, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeds = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        vector = (sum_embeds / sum_mask).squeeze().numpy()

    # Execution path for TensorFlow structures
    elif BACKEND == "TF":
        inputs = tokenizer(clean_text, max_length=45, padding=True, truncation=True, return_tensors="tf")
        if "token_type_ids" in inputs: del inputs["token_type_ids"]
        outputs = embedding_model(inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = tf.cast(inputs['attention_mask'], tf.float32)
        mask_expanded = tf.expand_dims(attention_mask, -1)
        sum_embeds = tf.reduce_sum(token_embeddings * mask_expanded, 1)
        sum_mask = tf.clip_by_value(tf.reduce_sum(mask_expanded, 1), 1e-9, True)
        vector = np.squeeze((sum_embeds / sum_mask).numpy())

    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

def load_routing_rules():
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {}

def start_decision_engine():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model matrix directory not found at: {MODEL_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    if BACKEND == "TORCH":
        full_model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, local_files_only=True)
        embedding_model = full_model.distilbert
    elif BACKEND == "TF":
        full_model = TFAutoModelForMaskedLM.from_pretrained(MODEL_PATH, local_files_only=True)
        embedding_model = full_model.distilbert
    
    # Secure absolute path resolution for database objects routing
    sys.path.insert(0, os.path.abspath(BASE_DIR))
    from data.base_knowledge import rhel_expert_matrix
    routing_rules = load_routing_rules()

    print("🧠 Indexing database tensors into RAM matrix...")
    vectors = np.array([get_mean_pooling_vector(item["intent"], tokenizer, embedding_model) for item in rhel_expert_matrix])

    print(f"\n🤖 RHEL ANALYTICAL SYSTEM READY [ENGINE BACKEND BOUNDARY: {BACKEND}].")
    
    while True:
        try: user_query = input("❓ Query: ")
        except (KeyboardInterrupt, EOFError): break
        if user_query.strip().lower() in {'exit', 'quit', 'q'}: break
        if not user_query.strip(): continue

        query_vec = get_mean_pooling_vector(user_query, tokenizer, embedding_model)
        scores = np.dot(vectors, query_vec)

        clean_query = user_query.lower()
        for target_keyword, anchor_tokens in routing_rules.items():
            if any(token in clean_query for token in anchor_tokens):
                for idx, item in enumerate(rhel_expert_matrix):
                    if target_keyword in item["solution"]: scores[idx] += 0.25

        best_idx = np.argmax(scores)
        confidence = min(100.0, scores[best_idx] * 100)
        matched_solution = rhel_expert_matrix[best_idx]['solution']

        print(f"\n======================================================================\n🎯 SEMANTIC MATCH  : {confidence:.1f}%\n💻 TARGET COMMAND  : {matched_solution}\n======================================================================\n")

if __name__ == "__main__":
    start_decision_engine()
