import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os

ours_dir = "experiments/public/all_scores/ours"
theirs = "experiments/public/all_scores/v2.0.1/VLM2Vec-V2.0-Qwen2VL-2B.json"

modality2metric = {
    "image": "hit@1",
    "video": "hit@1",
    "visdoc": "ndcg_linear@5",
}


with open(theirs, "r", encoding="utf-8") as f:
    theirs_scores = json.load(f)


for modality in theirs_scores["metrics"].keys():
    scores_dict = {
        "theirs": {
            d: theirs_scores["metrics"][modality][d][modality2metric[modality]]
            for d in theirs_scores["metrics"][modality].keys()
        }
    }
    for fname in os.listdir(ours_dir):
        if "avg" not in fname:
            ckpt_id = fname.replace(".json", "").split("_")[-1]
            with open(os.path.join(ours_dir, fname), "r", encoding="utf-8") as f:
                ours_ckpt_scores = json.load(f)

            scores_dict[ckpt_id] = {
                d: ours_ckpt_scores["metrics"][modality][d][modality2metric[modality]]
                for d in ours_ckpt_scores["metrics"][modality].keys()
            }

    df = pd.DataFrame.from_dict(scores_dict).fillna(0.0)
    print(df.columns)  # confirm columns exist & are named properly
    fig, ax = plt.subplots(figsize=(30, 15))
    df.plot(kind="bar", width=0.9, ax=ax)
    ax.set_title(f"Comparison of Models on {modality} Modality", fontsize=26)
    ax.set_xlabel("Datasets", fontsize=22)
    ax.set_ylabel(modality2metric[modality], fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(
        title="Models",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=18,
        title_fontsize=20,
    )
    plt.tight_layout()
    os.makedirs("experiments/public/plots", exist_ok=True)
    plt.savefig(f"experiments/public/plots/{modality}_comparison.png")
    plt.close()
