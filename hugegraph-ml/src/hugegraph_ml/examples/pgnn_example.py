import os
import warnings

import dgl

import numpy as np
import torch
import torch.nn as nn
from hugegraph_ml.models.pgnn import (
    PGNN,
    get_dataset,
    preselect_anchor,
    get_loss,
    train_model,
    eval_model,
)
from hugegraph_ml.data.hugegraph2dgl import HugeGraph2DGL

warnings.filterwarnings("ignore")


def pgnn_example(args):
    # The mean and standard deviation of the experiment results
    # are stored in the 'results' folder

    hg2d = HugeGraph2DGL()
    graph = hg2d.convert_graph_nx(
        vertex_label="CAVEMAN_vertex", edge_label="CAVEMAN_edge"
    )

    if not os.path.isdir("results"):
        os.mkdir("results")

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(
        "Learning Type: {}".format(["Transductive", "Inductive"][args.inductive]),
        "Task: {}".format(args.task),
    )

    results = []

    for repeat in range(args.repeat_num):
        data = get_dataset(args, graph)
        # pre-sample anchor nodes and compute shortest distance values for all epochs
        (
            g_list,
            anchor_eid_list,
            dist_max_list,
            edge_weight_list,
        ) = preselect_anchor(data, args)

        # model
        model = PGNN(input_dim=data["feature"].shape[1]).to(device)

        # loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        loss_func = nn.BCEWithLogitsLoss()

        best_auc_val = -1
        best_auc_test = -1

        for epoch in range(args.epoch_num):
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    param_group["lr"] /= 10

            g = dgl.graph(g_list[epoch])
            g.ndata["feat"] = torch.FloatTensor(data["feature"])
            g.edata["sp_dist"] = torch.FloatTensor(edge_weight_list[epoch])
            g_data = {
                "graph": g.to(device),
                "anchor_eid": anchor_eid_list[epoch],
                "dists_max": dist_max_list[epoch],
            }

            train_model(data, model, loss_func, optimizer, device, g_data)

            loss_train, auc_train, auc_val, auc_test = eval_model(
                data, g_data, model, loss_func, device
            )
            if auc_val > best_auc_val:
                best_auc_val = auc_val
                best_auc_test = auc_test

            if epoch % args.epoch_log == 0:
                print(
                    repeat,
                    epoch,
                    "Loss {:.4f}".format(loss_train),
                    "Train AUC: {:.4f}".format(auc_train),
                    "Val AUC: {:.4f}".format(auc_val),
                    "Test AUC: {:.4f}".format(auc_test),
                    "Best Val AUC: {:.4f}".format(best_auc_val),
                    "Best Test AUC: {:.4f}".format(best_auc_test),
                )

        results.append(best_auc_test)

    results = np.array(results)
    results_mean = np.mean(results).round(6)
    results_std = np.std(results).round(6)
    print("-----------------Final-------------------")
    print(results_mean, results_std)

    with open(
        "results/{}_{}_{}.txt".format(
            ["Transductive", "Inductive"][args.inductive],
            args.task,
            args.k_hop_dist,
        ),
        "w",
    ) as f:
        f.write("{}, {}\n".format(results_mean, results_std))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="link", choices=["link", "link_pair"]
    )
    parser.add_argument(
        "--inductive",
        action="store_true",
        help="Inductive learning or transductive learning",
    )
    parser.add_argument(
        "--k_hop_dist",
        default=-1,
        type=int,
        help="K-hop shortest path distance, -1 means exact shortest path.",
    )

    parser.add_argument("--epoch_num", type=int, default=200)
    parser.add_argument("--repeat_num", type=int, default=10)
    parser.add_argument("--epoch_log", type=int, default=100)

    args = parser.parse_args()
    pgnn_example(args)
