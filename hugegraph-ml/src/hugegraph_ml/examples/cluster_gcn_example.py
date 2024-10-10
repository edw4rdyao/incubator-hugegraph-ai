# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from hugegraph_ml.data.hugegraph2dgl import HugeGraph2DGL
from hugegraph_ml.models.cluster_gcn import SAGE
from hugegraph_ml.tasks.node_classify import NodeClassify

from typing import Literal
from hugegraph_ml.utils.early_stopping import EarlyStopping
import torch
import dgl
from torch import nn
import torchmetrics.functional as MF
from tqdm import trange
import numpy as np


def train_with_sample(
    model,
    graph,
    dataloader,
    lr: float = 1e-3,
    weight_decay: float = 0,
    n_epochs: int = 200,
    patience: int = float("inf"),
    early_stopping_monitor: Literal["loss", "accuracy"] = "loss",
    gpu: int = -1,
):
    # Set device for training
    device = f"cuda:{gpu}" if gpu != -1 and torch.cuda.is_available() else "cpu"
    early_stopping = EarlyStopping(patience=patience, monitor=early_stopping_monitor)
    model.to(device)
    # Get node features, labels, masks and move to device
    feats = graph.ndata["feat"].to(device)
    labels = graph.ndata["label"].to(device)
    train_mask = graph.ndata["train_mask"].to(device)
    val_mask = graph.ndata["val_mask"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Training model
    loss_fn = nn.CrossEntropyLoss()
    epochs = trange(n_epochs)
    for epoch in epochs:
        # train
        model.train()
        for it, sg in enumerate(dataloader):
            sg_feats = feats[sg.ndata["_ID"]]
            sg_labels = labels[sg.ndata["_ID"]]
            sg_train_msak = train_mask[sg.ndata["_ID"]].bool()
            logits = model(sg, sg_feats)
            train_loss = loss_fn(logits[sg_train_msak], sg_labels[sg_train_msak])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # validation
            valid_metrics = evaluate_sg(
                model=model, sg=sg, sg_feats=sg_feats, labels=labels, val_mask=val_mask
            )
            # logs
            epochs.set_description(
                f"epoch {epoch} | it {it} | train loss {train_loss.item():.4f} | val loss {valid_metrics['loss']:.4f}"
            )
            # early stopping
            early_stopping(valid_metrics[early_stopping.monitor], model)
            torch.cuda.empty_cache()
            if early_stopping.early_stop:
                break
        early_stopping.load_best_model(model)


def evaluate_sg(model, sg, sg_feats, labels, val_mask):
    model.eval()
    sg_val_msak = val_mask[sg.ndata["_ID"]].bool()
    sg_val_labels = labels[sg_val_msak]
    with torch.no_grad():
        sg_val_logits = model.inference(sg, sg_feats)[sg_val_msak]
        val_loss = model.loss(sg_val_logits, sg_val_labels)
        _, predicted = torch.max(sg_val_logits, dim=1)
        accuracy = (predicted == sg_val_labels).sum().item() / len(sg_val_labels)
    return {"accuracy": accuracy, "loss": val_loss.item()}


def evaluate(graph, model, dataloader):
    test_mask = graph.ndata["test_mask"]
    feats = graph.ndata["feat"]
    labels = graph.ndata["label"]
    test_logits = []
    test_labels = []
    total_loss = 0
    with torch.no_grad():
        for it, sg in enumerate(dataloader):
            sg_feats = feats[sg.ndata["_ID"]]
            sg_labels = labels[sg.ndata["_ID"]]
            sg_test_msak = test_mask[sg.ndata["_ID"]].bool()
            sg_test_labels = sg_labels[sg_test_msak]
            sg_test_logits = model.inference(sg, sg_feats)[sg_test_msak]
            loss = model.loss(sg_test_logits, sg_test_labels)
            total_loss += loss
            test_logits.append(sg_test_logits)
            test_labels.append(sg_test_labels)
        test_logits = torch.tensor(np.vstack(test_logits))
        _, predicted = torch.max(test_logits, dim=1)
        accuracy = (predicted == test_labels[0]).sum().item() / len(test_labels[0])
    return {"accuracy": accuracy, "total_loss": total_loss.item()}


def cluster_gcn_example(n_epochs=200):
    hg2d = HugeGraph2DGL()
    graph = hg2d.convert_graph(vertex_label="CORA_vertex", edge_label="CORA_edge")
    model = SAGE(
        in_feats=graph.ndata["feat"].shape[1],
        n_hidden=64,
        n_classes=graph.ndata["label"].unique().shape[0],
    )
    # sample
    num_partitions = 100
    sampler = dgl.dataloading.ClusterGCNSampler(
        graph,
        num_partitions,
        # prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )
    gpu = -1
    device = f"cuda:{gpu}" if gpu != -1 and torch.cuda.is_available() else "cpu"
    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(num_partitions).to(device),
        sampler,
        device=device,
        batch_size=100,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    train_with_sample(model=model, graph=graph, dataloader=dataloader)
    print(evaluate(graph=graph, model=model, dataloader=dataloader))


if __name__ == "__main__":
    cluster_gcn_example()
