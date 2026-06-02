# ============================================================
# Week 4 – Hierarchical Summarisation (Zero-Shot)
# IBM Granite 34B on HPC (2x A100)
# Group 13 – ARC, LIMBO, ACDC clusters
#
# Usage:
#   python week4_hpc_all.py --algo arc
#   python week4_hpc_all.py --algo limbo
#   python week4_hpc_all.py --algo acdc
# ============================================================

import os
import json
import csv
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 0. ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["arc", "limbo", "acdc"], required=True,
                    help="Which clustering algorithm's RSF to process")
args = parser.parse_args()
ALGO = args.algo

# ============================================================
# 1. CONFIGURATION — edit paths to match your HPC layout
# ============================================================
MODEL_NAME = "ibm-granite/granite-34b-code-instruct-8k"
HF_TOKEN   = os.environ.get("HF_TOKEN")

JAVA_ROOT  = os.path.expanduser("~/Week4/hadoop-mapreduce-client-core/src/main/java")

RSF_FILES = {
    "arc":   os.path.expanduser("~/Week4/MapReduce_ARC_jina_filtered.rsf"),
    "limbo": os.path.expanduser("~/Week4/mapreduce-3_6_0_IL_50_clusters.rsf"),
    "acdc":  os.path.expanduser("~/Week4/MapReduce_ACDC.rsf"),
}

OUTPUT_DIR = os.path.expanduser(f"~/Week4/outputs/{ALGO}_zeroshot")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE1_OUTPUT = os.path.join(OUTPUT_DIR, "file_summaries.json")
PHASE2_OUTPUT = os.path.join(OUTPUT_DIR, "cluster_descriptions.json")
CSV_OUTPUT    = os.path.join(OUTPUT_DIR, f"{ALGO}_results.csv")

RSF_PATH = RSF_FILES[ALGO]

print(f"\n{'='*60}")
print(f"  Algorithm : {ALGO.upper()}")
print(f"  RSF       : {RSF_PATH}")
print(f"  Output    : {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================
# 2. LOAD MODEL
# ============================================================
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, trust_remote_code=True
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model across 2x A100 GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded.\n")

# ============================================================
# 3. HELPER – INFERENCE
# ============================================================
def run_inference(messages, max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        messages,
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

    input_len = inputs["input_ids"].shape[1]
    response  = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    del inputs, outputs
    torch.cuda.empty_cache()
    return response.strip()

# ============================================================
# 4. PARSE RSF → cluster_to_files (FILE-LEVEL, preprocessed)
#
#   Preprocessing (required by TA for LIMBO and ACDC):
#   - Strip inner class suffixes:  Job$1 → Job
#   - Deduplicate: multiple inner classes of same file → one entry
#   - This ensures clusters contain .java file references, not class references
#
#   ARC already uses file-level entries but we apply the same
#   logic for consistency.
# ============================================================
print("Parsing RSF and applying file-level preprocessing...")

cluster_to_files = defaultdict(set)  # use set for automatic deduplication

with open(RSF_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3 or parts[0] != "contain":
            continue

        cluster_id = parts[1]   # string — works for both numeric (LIMBO/ARC) and named (ACDC)
        class_fqn  = parts[2]

        # KEY PREPROCESSING STEP:
        # Strip inner class suffix so Job$1, Job$2, Job$Builder all → Job
        base_fqn = class_fqn.split("$")[0]

        cluster_to_files[cluster_id].add(base_fqn)

# Convert sets to sorted lists
cluster_to_files = {k: sorted(v) for k, v in cluster_to_files.items()}

total_files   = sum(len(v) for v in cluster_to_files.values())
print(f"Clusters found : {len(cluster_to_files)}")
print(f"Unique files   : {total_files}\n")

# ============================================================
# 5. HELPER – FIND .java FILE FROM FQN
# ============================================================
def find_java_file(fqn):
    """Convert org.apache.hadoop.mapreduce.Job → /path/to/Job.java"""
    rel_path  = fqn.replace(".", "/") + ".java"
    full_path = os.path.join(JAVA_ROOT, rel_path)
    return full_path if os.path.exists(full_path) else None

# ============================================================
# 6. PHASE 1 – FILE-LEVEL SUMMARIES (Zero-Shot)
# ============================================================
print("="*60)
print("  PHASE 1 – File-level Summaries")
print("="*60)

# Resume support — skip already-summarised files
if os.path.exists(PHASE1_OUTPUT):
    with open(PHASE1_OUTPUT, "r") as f:
        file_summaries = json.load(f)
    print(f"Resuming — {len(file_summaries)} summaries already done.")
else:
    file_summaries = {}

# Collect all unique files across all clusters
all_files = sorted({fqn for files in cluster_to_files.values() for fqn in files})

for idx, fqn in enumerate(all_files):
    if fqn in file_summaries:
        continue

    java_file = find_java_file(fqn)
    if java_file is None:
        print(f"  ⚠ Not found: {fqn}")
        file_summaries[fqn] = "Source file not found."
        continue

    with open(java_file, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()

    # Truncate very large files
    if len(source_code) > 6000:
        source_code = source_code[:6000] + "\n// [TRUNCATED]"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a software architecture expert. "
                "Analyze Java source code and extract a concise semantic summary."
            )
        },
        {
            "role": "user",
            "content": (
                "Analyze the following Java source code and provide a structured summary covering:\n"
                "1. Key functionality\n"
                "2. Core logic\n"
                "3. Inputs and Outputs\n"
                "4. Dependencies (other classes/packages it relies on)\n\n"
                "Keep the summary concise (maximum 100 words).\n\n"
                f"<source_code>\n{source_code}\n</source_code>"
            )
        }
    ]

    summary = run_inference(messages, max_new_tokens=256)
    file_summaries[fqn] = summary
    print(f"  [{idx+1}/{len(all_files)}] ✓ {fqn.split('.')[-1]}")

    # Save after every file (crash recovery)
    with open(PHASE1_OUTPUT, "w") as f:
        json.dump(file_summaries, f, indent=2)

print(f"\n✅ Phase 1 complete — {len(file_summaries)} summaries saved to {PHASE1_OUTPUT}\n")

# ============================================================
# 7. PHASE 2 – CLUSTER-LEVEL DESCRIPTIONS (Zero-Shot)
#
#   Updated prompt explicitly asks for:
#   - Components and interactions
#   - Quality attributes (scalability, maintainability, security, etc.)
#   - Technologies and frameworks used
#   - Strict 150-word limit
# ============================================================
print("="*60)
print("  PHASE 2 – Cluster-level Descriptions")
print("="*60)

cluster_descriptions = {}

for cluster_id, files in cluster_to_files.items():
    print(f"\nCluster {cluster_id} ({len(files)} files)...")

    # Build summaries block
    summaries_text = ""
    for fqn in files:
        short_name    = fqn.split(".")[-1]
        summary       = file_summaries.get(fqn, "No summary available.")
        summaries_text += f"\n### {short_name}\n{summary}\n"

    # Truncate if too long for context window
    if len(summaries_text) > 8000:
        summaries_text = summaries_text[:8000] + "\n... [TRUNCATED]"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a software architecture expert specializing in recovering "
                "architectural descriptions from source code summaries."
            )
        },
        {
            "role": "user",
            "content": (
                f"The following are semantic summaries of Java classes belonging to "
                f"the same architectural cluster (Cluster {cluster_id}) in Apache Hadoop MapReduce.\n\n"
                f"{summaries_text}\n\n"
                "Based on these summaries, provide:\n"
                "1. A short architectural title (3-6 words)\n"
                "2. A concise description (strictly under 150 words) that covers:\n"
                "   - The main components and how they interact with each other\n"
                "   - Quality attributes achieved by this cluster "
                "(e.g. scalability, maintainability, fault-tolerance, security)\n"
                "   - Technologies and frameworks used "
                "(e.g. Java, Hadoop, YARN, Avro, Protocol Buffers)\n"
                "   - The role this cluster plays in the overall MapReduce system\n\n"
                "Format your response exactly as:\n"
                "Title: <your title>\n"
                "Description: <your description>"
            )
        }
    ]

    raw_output = run_inference(messages, max_new_tokens=300)

    # Parse title and description from model output
    title       = ""
    description = ""
    for line in raw_output.splitlines():
        if line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
        elif line.startswith("Description:"):
            description = line.replace("Description:", "").strip()

    # Fallback: if model didn't follow format, use full output as description
    if not title and not description:
        title       = f"Cluster {cluster_id}"
        description = raw_output.strip()

    cluster_descriptions[str(cluster_id)] = {
        "files":       files,
        "title":       title,
        "description": description,
        "raw_output":  raw_output  # keep for debugging
    }

    print(f"  Title : {title}")
    print(f"  Desc  : {description[:100]}...")

    # Save after every cluster
    with open(PHASE2_OUTPUT, "w") as f:
        json.dump(cluster_descriptions, f, indent=2)

print(f"\n✅ Phase 2 complete — saved to {PHASE2_OUTPUT}\n")

# ============================================================
# 8. EXPORT CSV (required submission format)
#
#   Columns: cluster_ID | files | title | description
# ============================================================
print("="*60)
print("  Exporting CSV")
print("="*60)

with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["cluster_ID", "files", "title", "description"])
    writer.writeheader()
    for cluster_id, data in cluster_descriptions.items():
        writer.writerow({
            "cluster_ID":  cluster_id,
            "files":       "; ".join(data["files"]),  # semicolon-separated list
            "title":       data["title"],
            "description": data["description"]
        })

print(f"✅ CSV saved to {CSV_OUTPUT}")
print(f"\n🎉 All done for {ALGO.upper()}!")
print(f"   Phase 1 summaries : {PHASE1_OUTPUT}")
print(f"   Phase 2 JSON      : {PHASE2_OUTPUT}")
print(f"   CSV (submission)  : {CSV_OUTPUT}")
