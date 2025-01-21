import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

estimators = [
    "UniversalUnconstrained",
    "UniversalConstrained",
    "PermutationSHAP",
    # "KernelBanzhaf",
    "LeverageSHAP",
    "KernelSHAP",
    # "MonteCarlo",
    # "MSR",
    "OFA_A",
    "OFA_S",
    "WSL"
]
estimators.sort(key=len, reverse=True)

prob_dict = {
    "beta_shapley_1_16": "beta_shapley_1_16",
    "beta_shapley_1_4": "beta_shapley_1_4",
    "shapley": "beta_shapley_1_1",
    "beta_shapley_4_1": "beta_shapley_4_1",
    "beta_shapley_16_1": "beta_shapley_16_1",
    "weighted_banzhaf_0.1": "weighted_banzhaf_0.1",
    "weighted_banzhaf_0.3": "weighted_banzhaf_0.3",
    "banzhaf": "weighted_banzhaf_0.5",
    "weighted_banzhaf_0.7": "weighted_banzhaf_0.7",
    "weighted_banzhaf_0.9": "weighted_banzhaf_0.9",
    "random_42": "random_42",
    "random_27": "random_27",
    "random_91": "random_91",
    "random_58": "random_58",
    "random_16": "random_16",
}
prob_values = [v for k, v in prob_dict.items()]
print(prob_values)

dataset = "California"

random = False
if not random:
    prob_values = [pv for pv in prob_values if "random" not in pv]

def parse_filename(fname):
    if fname.startswith(f"{dataset}_"):
        base = fname[len(f"{dataset}_"):]
    else:
        base = fname

    if base.endswith(".csv"):
        base = base[:-4]

    chosen_estimator = None
    for est in estimators:
        if est in base:
            chosen_estimator = est
            base = base.replace(est, "", 1).strip("_")
            break
    
    chosen_prob_value = None
    if "weighted" in base or "beta" in base:
        enumerate_list = prob_dict.keys() - ["shapley", "banzhaf"]
    else:
        enumerate_list = prob_dict.keys()
    for pv in enumerate_list:
        if pv in base:
            chosen_prob_value = prob_dict[pv]
            base = base.replace(pv, "", 1).strip("_")
            break
    
    return chosen_estimator, chosen_prob_value

def parse_line(line):
    line = line.strip()
    if not line:
        return None
    
    line = line.replace("np.float64(", "")
    while "np.float64(" in line:
        line = line.replace("np.float64(", "")
    line = line.replace(")", "")
    
    line = line.replace("'", '"')
    #  {"sample_size": 72, "difference": 0, "noise": 0, "n": 12, "error": 1.2345e-5, "sum_error": 1.2345e-4}
    
    dct = json.loads(line)
    return dct

records = []

csv_files = [os.path.join("./output", fname) for fname in os.listdir("./output") if fname.endswith(".csv")]

for fname in csv_files:
    if dataset not in fname:
        continue
    estimator, prob_value = parse_filename(fname)
    if estimator is None or prob_value is None:
        print(f"Skipping unknown filename: {fname}")
        continue
    
    with open(fname, "r") as fin:
        for line in fin:
            data = parse_line(line)
            if not data:
                continue
            
            sample_size = data["sample_size"]
            error_val = data["error"]
            records.append({
                "estimator": estimator,
                "prob_value": prob_value,
                "sample_size": sample_size,
                "error": error_val,
            })

df = pd.DataFrame(records)
print("Data loaded:", df.shape)

grouped = df.groupby(["estimator", "prob_value", "sample_size"])["error"]
stats_df = grouped.agg(
    median_error = "median",
    q25         = lambda x: np.percentile(x, 25),
    q75         = lambda x: np.percentile(x, 75),
).reset_index()

sns.set_style("whitegrid")

if random:
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(17, 10), sharex=False, sharey=False)
else:
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(17, 7), sharex=False, sharey=False)
axes = axes.ravel()

estimator_list = sorted(df["estimator"].unique())
colors = sns.color_palette("tab10", n_colors=len(estimator_list))

for i, pv in enumerate(prob_values):
    ax = axes[i]
    
    sub_df = stats_df[stats_df["prob_value"] == pv]
    if sub_df.empty:
        ax.set_title(f"{pv} (no data)")
        continue
    
    for c_idx, est in enumerate(estimator_list):
        est_data = sub_df[sub_df["estimator"] == est].sort_values("sample_size")
        if est_data.empty:
            continue
        
        x = est_data["sample_size"]
        y_median = est_data["median_error"]
        y_lower = est_data["q25"]
        y_upper = est_data["q75"]
        
        ax.plot(x, y_median, label=est, color=colors[c_idx])
        ax.fill_between(x, y_lower, y_upper, color=colors[c_idx], alpha=0.2)

    ax.set_title(pv)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("L2 Error")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(False)

plt.tight_layout()


all_handles = []
all_labels = []

for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    all_handles += handles
    all_labels += labels
by_label = dict(zip(all_labels, all_handles))
fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4)

plt.subplots_adjust(bottom=0.15)
plt.savefig(f"{dataset}_sample_size.png")
