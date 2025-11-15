import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict

base_dir = "/lambda/nfs/poria-cvpr-2026/varun/vlm2vec2/exps/Qwen2vl_2B.image+visdoc+video.autoresize.lora1.BS1024.IB64.GCq8p8.NormTemp002.lr5e-5.step5kwarm100.auxenc.parallel.hidden512.layers28.gqa.attnqknorm.heads8.kvheads4.intsize2048/checkpoint-2000"
taxonomy_path = "experiments/public/eval/mmeb_v2_taxonomy.json"
out_dir = "experiments/public/all_scores/ours"
model_name = "qwen2_vl"
modality2metric = {
    "image": "hit@1",
    "video": "hit@1",
    "visdoc": "ndcg_linear@5",
}

ckpt_id = base_dir.split("/")[-1].split("-")[-1]
os.makedirs(out_dir, exist_ok=True)

full_data = {
    "metadata": {
        "model_name": model_name,
        "ckpt_id": ckpt_id,
        "timestamp": datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
    },
    "metrics": {},
}

for modality in os.listdir(base_dir):
    modality_path = os.path.join(base_dir, modality)
    full_data["metrics"][modality] = {}

    for fname in os.listdir(modality_path):
        if fname.endswith("_score.json"):
            score_path = os.path.join(modality_path, fname)
            ds_name = fname.replace("_score.json", "")
            with open(score_path, "r", encoding="utf-8") as f:
                full_data["metrics"][modality][ds_name] = json.load(f)

with open(
    os.path.join(out_dir, f"{model_name}_ckpt-{ckpt_id}.json"), "w", encoding="utf-8"
) as f:
    json.dump(full_data, f, indent=2)

with open(taxonomy_path, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

average_scores = defaultdict(dict)

for modality, categories in taxonomy.items():
    for category, ds_names in categories.items():
        try:
            average_scores[modality][category] = np.mean(
                [
                    full_data["metrics"][modality][d][modality2metric[modality]]
                    for d in ds_names
                ]
            )
        except KeyError:
            pass

with open(
    os.path.join(out_dir, f"{model_name}_ckpt-{ckpt_id}_avg_scores.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(average_scores, f, indent=2)
