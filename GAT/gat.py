

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import warnings
from GAT.Layers.GAT_Layers.GraphAttentionNetwork import GraphAttentionNetwork
from GAT.Layers.GATV2_Layers.GraphAttentionNetworkV2 import GraphAttentionNetworkV2
from embeddings.data import datasets
from matplotlib import pyplot as plt
from tensorflow.keras import utils


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)

"""
## Obtain the dataset
"""


def datasetw(dataset):
    g = dataset.load()
    g.gatdata()
    citations = g.citations
    papers = g.papers

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

    """
    ### Split the dataset
    """

    # Obtain random indices
    random_indices = np.random.permutation(range(papers.shape[0]))

    # 50/50 split
    train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
    test_data = papers.iloc[random_indices[len(random_indices) // 2:]]


    """
    ### Prepare the graph data
    """

    # Obtain paper indices which will be used to gather node states
    # from the graph later on when training the model
    all_indices = papers["paper_id"].to_numpy()
    train_indices = train_data["paper_id"].to_numpy()
    test_indices = test_data["paper_id"].to_numpy()

    # Obtain ground truth labels corresponding to each paper_id
    train_labels = train_data["subject"].to_numpy()
    test_labels = test_data["subject"].to_numpy()
    all_labels = papers["subject"].to_numpy()

    # Define graph, namely an edge tensor and a node feature tensor
    edges = tf.convert_to_tensor(citations[["target", "source"]])
    node_states = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])


    return class_values, node_states, edges, train_indices, test_indices, test_labels, train_labels, all_indices, all_labels



"""
### Train and evaluate
"""


def train(dataset, gatv2):
    class_values, node_states, edges, train_indices, test_indices, test_labels, train_labels, all_indices, all_labels = datasetw(dataset)

    # Define hyper-parameters
    HIDDEN_UNITS = 100
    NUM_HEADS = 8
    NUM_LAYERS = 3
    OUTPUT_DIM = len(class_values)
    OUTPUT_DIM = 16

    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 3e-1
    MOMENTUM = 0.9

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
    )

    # Build model
    if gatv2:
        gat_model = GraphAttentionNetworkV2(
            node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
        )
    else:
        gat_model = GraphAttentionNetwork(
            node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
        )

    # Compile model
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])


    history = gat_model.fit(
        x=train_indices,
        y=train_labels,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )



    # plot loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # plot acc
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    _, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)

    print("--" * 38 + f"\nTest Accuracy {test_accuracy * 100:.1f}%")

    embeddings = gat_model.predict(all_indices)

    return embeddings



