import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


JSON_FILE = "/eos/user/s/swaldych/smart_pix/labels/models/xgboost_model_50x12P5x150_0fb_0P2thresh.json"
NUM_CLASS = 3

with open(JSON_FILE, "r") as f:
    data = json.load(f)

# Pull trees from the save_model JSON structure
trees = data["learner"]["gradient_booster"]["model"]["trees"]
num_trees = len(trees)

# best_iteration is stored as a string in attributes
attrs = data["learner"].get("attributes", {})
best_iter = int(attrs.get("best_iteration", "-1"))
rounds = best_iter + 1 if best_iter >= 0 else None

# Count nodes per tree in save_model format:
# Each node index corresponds to one entry in left_children/right_children arrays.
def nodes_in_tree(tree):
    if "left_children" in tree:
        return len(tree["left_children"])
    # fallback to other per-node arrays if needed
    for k in ["right_children", "split_indices", "split_conditions", "base_weights"]:
        if k in tree:
            return len(tree[k])
    raise KeyError(f"Can't infer node count from keys: {list(tree.keys())}")

nodes_per_tree = [nodes_in_tree(t) for t in trees]
total_nodes = sum(nodes_per_tree)

print("File:", JSON_FILE)
print("Total trees:", num_trees)

if rounds is not None:
    print("best_iteration:", best_iter)
    print("Boosting rounds:", rounds)
    print("num_class:", NUM_CLASS)
    print("Expected trees (rounds * num_class):", rounds * NUM_CLASS)

print("\nNode counts:")
print("Total nodes:", total_nodes)
print("Average nodes/tree:", total_nodes / num_trees)
print("Max nodes in one tree:", max(nodes_per_tree))
print("Min nodes in one tree:", min(nodes_per_tree))

#---------------------------------------------------

pred_NN = pd.read_csv('/eos/user/s/swaldych/smart_pix/labels/results/pred_class_NN_50x12P5x150_0fb_0P2thresh.csv')
pred_BDT= pd.read_csv('/eos/user/s/swaldych/smart_pix/labels/results/pred_class_BDT_50x12P5x150_0fb_0P2thresh.csv')

pt_test_NN = pd.read_csv('/eos/user/s/swaldych/smart_pix/labels/results/pt_test_NN_50x12P5x150_0fb_0P2thresh.csv')
pt_test_BDT= pd.read_csv('/eos/user/s/swaldych/smart_pix/labels/results/pt_test_BDT_50x12P5x150_0fb_0P2thresh.csv')

pt_test_NN= np.asarray(pt_test_NN).ravel().astype(float)
pt_test_BDT= np.asarray(pt_test_BDT).ravel().astype(float)

accepted_NN  = (pred_NN == 0)
accepted_BDT = (pred_BDT == 0)

def acceptance_vs_pt(pt_test, accepted, step=0.2):

    pt_test = np.asarray(pt_test).ravel().astype(float)
    accepted = np.asarray(accepted).ravel().astype(bool)

    pt_vals, acc_vals, err_vals = [], [], []

    pmin = float(np.min(pt_test))
    pmax = float(np.max(pt_test))

    p = pmin
    while p < pmax:

        mask = (pt_test >= p) & (pt_test < p + step)
        total = int(np.sum(mask))

        if total > 0:
            passed = int(np.sum(accepted[mask]))
            acc = passed / total

            pt_vals.append(p + step/2)
            acc_vals.append(acc)
            err_vals.append(np.sqrt(acc * (1 - acc) / total))

        p += step

    return np.array(pt_vals), np.array(acc_vals), np.array(err_vals)

pt_NN, acc_NN, err_NN = acceptance_vs_pt(pt_test_NN, accepted_NN)
pt_BDT, acc_BDT, err_BDT = acceptance_vs_pt(pt_test_BDT, accepted_BDT)

plt.errorbar(pt_NN, acc_NN, err_NN, fmt='o', markersize=3, label='NN')
plt.errorbar(pt_BDT, acc_BDT, err_BDT, fmt='s', markersize=3, label='BDT')

plt.xlabel("true pT (GeV)")
plt.ylabel("classifier acceptance")
plt.title("NN vs BDT acceptance vs pT")
plt.ylim(0,1)
plt.legend()

plt.savefig("/eos/user/s/swaldych/smart_pix/labels/models/NN_BDT_class_acceptance.png",
            dpi=300, bbox_inches="tight")
plt.show()