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
from hugegraph_ml.models.jknet import JKNet
from hugegraph_ml.tasks.node_classify import NodeClassify


def jknet_example():
    g2d = HugeGraph2DGL()
    graph, graph_info = g2d.convert_graph(
        vertex_label="cora_vertex", edge_label="cora_edge", info_vertex_label="cora_info_vertex"
    )
    model = JKNet(
        n_in_feats=graph_info["n_feat_dim"],
        n_out_feats=graph_info["n_classes"],
        mode="max"
    )
    node_clf_task = NodeClassify(graph, graph_info, model)
    node_clf_task.train(lr=0.005, weight_decay=0.0005, n_epochs=200, patience=200)
    print(node_clf_task.evaluate())


if __name__ == "__main__":
    jknet_example()
