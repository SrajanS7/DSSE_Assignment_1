# ==========================================
# Week 4 – Few-Shot Prompting
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
OUTPUT_DIR    = os.path.expanduser("~/Week4/outputs/fewshot")
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
def run_inference(prompt_messages, max_new_tokens=512):
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
# FEW-SHOT EXAMPLES (used in both phases)
# ==========================================
FILE_EXAMPLE_INPUT = """public class JobID extends ID {
    private final int jobId;
    public JobID(String jtIdentifier, int id) {
        this.jobId = id;
    }
    public int getId() { return jobId; }
}"""

FILE_EXAMPLE_OUTPUT = """1. Key functionality: Represents a unique identifier for a MapReduce Job.
2. Core logic: Extends the base ID class, storing a job-specific integer ID along with a JobTracker identifier.
3. Inputs/Outputs: Takes a JobTracker identifier string and integer ID as input; provides getId() to retrieve the job ID.
4. Dependencies: Inherits from org.apache.hadoop.mapreduce.ID."""

CLUSTER_EXAMPLE_INPUT = """### JobID
Represents a unique identifier for a MapReduce Job. Extends ID class with job-specific integer.
### TaskID
Represents a unique identifier for a MapReduce Task. Extends ID, references parent JobID.
### TaskAttemptID
Represents a unique identifier for a task attempt. Extends TaskID with attempt number."""

CLUSTER_EXAMPLE_OUTPUT = """Title: Job and Task Identity Management
Description: This cluster manages the identity and naming scheme for MapReduce execution units. It provides a hierarchy of unique identifiers — Job, Task, and TaskAttempt — that track execution at each level of the MapReduce lifecycle. The classes follow an inheritance pattern, each extending a base ID class and adding specificity. They serve as the primary reference keys used across the MapReduce system for tracking, logging, and coordinating distributed computation."""

# ==========================================
# 5. PHASE 1 – FEW-SHOT FILE SUMMARIES
# ==========================================
print("\n" + "="*60)
print("  PHASE 1 – Few-Shot File-level Summaries")
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

        if len(source_code) > 6000:
            source_code = source_code[:6000] + "\n// [TRUNCATED]"

        # FEW-SHOT PROMPT — one example provided before the actual request
        prompt = [
            {
                "role": "system",
                "content": "You are a software architecture expert. Analyze Java source code and extract a concise semantic summary. Follow the format shown in the example."
            },
            {
                "role": "user",
                "content": f"""Here is an example of the analysis I need:

Example Input:
<source_code>
{FILE_EXAMPLE_INPUT}
</source_code>

Example Output:
{FILE_EXAMPLE_OUTPUT}

Now analyze the following Java source code in the same format:
1. Key functionality
2. Core logic
3. Inputs and Outputs
4. Dependencies

Keep the summary concise (maximum 100 words).

<source_code>
{source_code}
</source_code>"""
            }
        ]

        summary = run_inference(prompt, max_new_tokens=256)
        file_summaries[fqn] = summary
        print(f"  ✅ {fqn.split('.')[-1]}")

        with open(PHASE1_OUTPUT, "w") as f:
            json.dump(file_summaries, f, indent=2)

print(f"\n✅ Phase 1 complete. Saved to {PHASE1_OUTPUT}")

# ==========================================
# 6. PHASE 2 – FEW-SHOT CLUSTER DESCRIPTIONS
# ==========================================
print("\n" + "="*60)
print("  PHASE 2 – Few-Shot Cluster-level Descriptions")
print("="*60)

cluster_descriptions = {}

for cluster_id, classes in cluster_to_classes.items():
    print(f"\nProcessing Cluster {cluster_id} ({len(classes)} classes)...")

    summaries_text = ""
    for fqn in classes:
        short_name = fqn.split(".")[-1]
        summary    = file_summaries.get(fqn, "No summary available.")
        summaries_text += f"\n### {short_name}\n{summary}\n"

    if len(summaries_text) > 8000:
        summaries_text = summaries_text[:8000] + "\n... [TRUNCATED]"

    # FEW-SHOT PROMPT — one example provided before the actual request
    prompt = [
        {
            "role": "system",
            "content": "You are a software architecture expert. Generate architectural titles and descriptions from class summaries. Follow the format shown in the example."
        },
        {
            "role": "user",
            "content": f"""Here is an example of the analysis I need:

Example Input:
{CLUSTER_EXAMPLE_INPUT}

Example Output:
{CLUSTER_EXAMPLE_OUTPUT}

Now analyze the following class summaries from Cluster {cluster_id} in Apache Hadoop MapReduce in the same format.
Provide:
1. A short architectural title (3-6 words)
2. A high-level descriptive summary (3-5 sentences) explaining:
   - The overall behaviour of this cluster
   - The architecture and design patterns used
   - How the components interact with each other
   - The role this cluster plays in the MapReduce system

Class summaries:
{summaries_text}"""
        }
    ]

    description = run_inference(prompt, max_new_tokens=512)
    cluster_descriptions[str(cluster_id)] = {
        "classes": classes,
        "num_classes": len(classes),
        "description": description
    }
    print(f"  → {description[:150]}...")

    with open(PHASE2_OUTPUT, "w") as f:
        json.dump(cluster_descriptions, f, indent=2)

print(f"\n✅ Phase 2 complete. Saved to {PHASE2_OUTPUT}")
print(f"\n🎉 Few-Shot complete! Outputs in: {OUTPUT_DIR}")
