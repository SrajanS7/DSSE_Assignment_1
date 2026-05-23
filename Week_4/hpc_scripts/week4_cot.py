# ==========================================
# Week 4 – Chain-of-Thought Prompting
# IBM Granite 34B on HPC (2x A100)
# Group 13 – MapReduce ARC Clusters (Jina)
# ==========================================

import os
import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 0. CONFIGURATION
# ==========================================
model_name    = "ibm-granite/granite-34b-code-instruct-8k"
hf_token      = os.environ.get('HF_TOKEN')

RSF_PATH      = os.path.expanduser("~/Week4/MapReduce_ARC_jina_filtered.rsf")
JAVA_ROOT     = os.path.expanduser("~/Week4/hadoop-mapreduce-client-core/src/main/java")
OUTPUT_DIR    = os.path.expanduser("~/Week4/outputs/cot")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE1_OUTPUT = os.path.join(OUTPUT_DIR, "file_summaries.json")
PHASE2_OUTPUT = os.path.join(OUTPUT_DIR, "cluster_descriptions.json")

# ==========================================
# 1. LOAD MODEL
# ==========================================
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model across 2x A100 GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model loaded.")

# ==========================================
# 2. HELPER – RUN INFERENCE
# ==========================================
def run_inference(prompt_messages, max_new_tokens=2048):
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )
    del inputs, outputs
    torch.cuda.empty_cache()
    return response

# ==========================================
# 3. PARSE RSF → MAP CLASS TO CLUSTER
# ==========================================
print("Parsing RSF file...")
class_to_cluster = {}
with open(RSF_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "contain":
            cluster_id = int(parts[1])
            class_name = parts[2]
            class_to_cluster[class_name] = cluster_id

cluster_to_classes = defaultdict(list)
for cls, cid in class_to_cluster.items():
    cluster_to_classes[cid].append(cls)

print(f"Total clusters: {len(cluster_to_classes)}")
print(f"Total classes:  {len(class_to_cluster)}")

# ==========================================
# 4. HELPER – FIND JAVA FILE FROM CLASS NAME
# ==========================================
def find_java_file(fqn, java_root):
    base      = fqn.split("$")[0]
    rel_path  = base.replace(".", "/") + ".java"
    full_path = os.path.join(java_root, rel_path)
    if os.path.exists(full_path):
        return full_path
    return None

# ==========================================
# 5. CLEAN BAD ENTRIES FROM PREVIOUS RUN
# ==========================================
if os.path.exists(PHASE2_OUTPUT):
    with open(PHASE2_OUTPUT, "r") as f:
        existing = json.load(f)
    cleaned = {k: v for k, v in existing.items()
               if v["description"].strip() != "... [TRUNCATED]"
               and len(v["description"].strip()) >= 50}
    removed = len(existing) - len(cleaned)
    if removed > 0:
        print(f"🧹 Removed {removed} bad entries from previous run.")
        with open(PHASE2_OUTPUT, "w") as f:
            json.dump(cleaned, f, indent=2)
    else:
        print(f"✅ No bad entries found in existing output.")

# ==========================================
# 6. PHASE 1 – COT FILE SUMMARIES
# ==========================================
print("\n" + "="*60)
print("  PHASE 1 – Chain-of-Thought File-level Summaries")
print("="*60)

if os.path.exists(PHASE1_OUTPUT):
    with open(PHASE1_OUTPUT, "r") as f:
        file_summaries = json.load(f)
    print(f"Resuming — {len(file_summaries)} summaries already done.")
else:
    file_summaries = {}

for cluster_id, classes in cluster_to_classes.items():
    print(f"\nCluster {cluster_id} ({len(classes)} classes)...")
    for fqn in classes:
        if fqn in file_summaries:
            continue

        java_file = find_java_file(fqn, JAVA_ROOT)
        if java_file is None:
            print(f"  ⚠️  File not found: {fqn}")
            file_summaries[fqn] = "File not found."
            continue

        with open(java_file, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

        # Silently truncate — no tag added
        if len(source_code) > 6000:
            source_code = source_code[:6000]

        prompt = [
            {
                "role": "system",
                "content": "You are a software architecture expert. Think step by step when analyzing Java source code. Show your reasoning at each step before writing the final summary."
            },
            {
                "role": "user",
                "content": f"""Analyze the following Java source code by thinking through it step by step:

Step 1 - Class purpose: What is the name of this class and what is its primary responsibility?
Step 2 - Core logic: Walk through the main methods. What algorithms or processes are implemented?
Step 3 - Inputs and Outputs: What data does this class receive (constructor args, method params) and what does it produce or return?
Step 4 - Dependencies: What other classes, interfaces, or packages does this class import or extend?
Step 5 - Final summary: Based on your reasoning above, write a concise summary (max 100 words) of this class's architectural role.

<source_code>
{source_code}
</source_code>"""
            }
        ]

        summary = run_inference(prompt, max_new_tokens=768)
        file_summaries[fqn] = summary
        print(f"  ✅ {fqn.split('.')[-1]}")

        with open(PHASE1_OUTPUT, "w") as f:
            json.dump(file_summaries, f, indent=2)

print(f"\n✅ Phase 1 complete. Saved to {PHASE1_OUTPUT}")

# ==========================================
# 7. PHASE 2 – COT CLUSTER DESCRIPTIONS
# ==========================================
print("\n" + "="*60)
print("  PHASE 2 – Chain-of-Thought Cluster-level Descriptions")
print("="*60)

if os.path.exists(PHASE2_OUTPUT):
    with open(PHASE2_OUTPUT, "r") as f:
        cluster_descriptions = json.load(f)
    print(f"Resuming — {len(cluster_descriptions)} clusters already done.")
else:
    cluster_descriptions = {}

for cluster_id, classes in cluster_to_classes.items():
    if str(cluster_id) in cluster_descriptions:
        print(f"  Skipping Cluster {cluster_id} (already done)")
        continue

    print(f"\nProcessing Cluster {cluster_id} ({len(classes)} classes)...")

    summaries_text = ""
    for fqn in classes:
        short_name = fqn.split(".")[-1]
        summary    = file_summaries.get(fqn, "No summary available.")
        summaries_text += f"\n### {short_name}\n{summary}\n"

    # Silently truncate — no tag added
    if len(summaries_text) > 8000:
        summaries_text = summaries_text[:8000]

    prompt = [
        {
            "role": "system",
            "content": "You are a software architecture expert. Think step by step when recovering architectural descriptions from source code summaries. Show your reasoning before writing the final output."
        },
        {
            "role": "user",
            "content": f"""You are given semantic summaries of Java classes belonging to Cluster {cluster_id} in Apache Hadoop MapReduce.
Reason through this step by step:

Step 1 - Common themes: What functionality do these classes share? What domain do they belong to?
Step 2 - Interactions: How do these classes depend on or communicate with each other? Are there clear caller/callee relationships?
Step 3 - Design patterns: What architectural or design patterns are visible? (e.g. Factory, Strategy, Observer, Template Method)
Step 4 - System role: What role does this cluster play in the overall MapReduce execution pipeline?
Step 5 - Final output: Based on your reasoning above, provide:
   a) A short architectural title (3-6 words)
   b) A high-level descriptive summary (3-5 sentences) covering overall behaviour, architecture, component interactions, and role in MapReduce.

Class summaries:
{summaries_text}"""
        }
    ]

    description = run_inference(prompt, max_new_tokens=2048)
    cluster_descriptions[str(cluster_id)] = {
        "classes": classes,
        "num_classes": len(classes),
        "description": description
    }
    print(f"  → {description[:150]}...")

    with open(PHASE2_OUTPUT, "w") as f:
        json.dump(cluster_descriptions, f, indent=2)

print(f"\n✅ Phase 2 complete. Saved to {PHASE2_OUTPUT}")
print(f"\n🎉 Chain-of-Thought complete! Outputs in: {OUTPUT_DIR}")
