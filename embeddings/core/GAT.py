import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from embeddings.data import datasets
from tensorflow.keras import optimizers
from embeddings.utils import split_data, convert_to_numeric_arrays, evaluate_preds, evaluation



import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)


def cora2():
    data_dir = "C:/Users/mehdi/PycharmProjects/graph_embeddings_pro/embeddings/GAT2/data/cora"

    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )

    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"),
        sep="\t",
        header=None,
        names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
    )

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

    #split dataset
    # Obtain random indices
    random_indices = np.random.permutation(range(papers.shape[0]))

    # 50/50 split
    train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
    test_data = papers.iloc[random_indices[len(random_indices) // 2:]]

    #Prepare the graph data
    # Obtain paper indices which will be used to gather node states
    # from the graph later on when training the model
    train_indices = train_data["paper_id"].to_numpy()
    test_indices = test_data["paper_id"].to_numpy()


    # Obtain ground truth labels corresponding to each paper_id
    train_labels = train_data["subject"].to_numpy()
    test_labels = test_data["subject"].to_numpy()

    # Define graph, namely an edge tensor and a node feature tensor
    edges = tf.convert_to_tensor(citations[["target", "source"]])
    node_states = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

    # Print shapes of the graph
    print("Edges shape:\t\t", edges.shape)
    print("Node features shape:", node_states.shape)

    # Define hyper-parameters
    HIDDEN_UNITS = 100
    NUM_HEADS = 8
    NUM_LAYERS = 3
    OUTPUT_DIM = len(class_values)

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
    gat_model = GraphAttentionNetwork(
        node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
    )

    # Compile model
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])
    #gat_model.compile(loss=loss_fn, optimizer=optimizers.Adam(lr=0.01), metrics=[accuracy_fn])

    gat_model.fit(
        x=train_indices,
        y=train_labels,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )

    _, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)

    print("--" * 38 + f"\nTest Accuracy {test_accuracy * 100:.1f}%")



class GraphAttention(layers.Layer):
    def __init__(
            self,
            units,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out




class GraphAttention2(layers.Layer):
    def __init__(
            self,
            units,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        #attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))

        x = tf.nn.leaky_relu(node_states_expanded)
        attention_scores = tf.matmul(x, self.kernel_attention)
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)


    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        #######
        LEARNING_RATE = 3e-1
        MOMENTUM = 0.9
        optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)

        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_states, self.edges])
            # Compute loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}


def cora():
    dataset = datasets.Cora()
    g = dataset.load()

    features = g.features.to_numpy()


'''
    #Split the dataset
    # Obtain random indices
    random_indices = np.random.permutation(range(papers.shape[0]))

    # 50/50 split
    train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
    test_data = papers.iloc[random_indices[len(random_indices) // 2:]]

    print(train_data)




    n_labels = g.get_number_of_subjects()
    node_subjects = g.get_node_subjects()

    train_subjects, val_subjects, test_subjects = split_data(node_subjects, 800, 900)
    train_targets, val_targets, test_targets = convert_to_numeric_arrays(train_subjects, val_subjects, test_subjects)



'''




if __name__ == '__main__':
   cora2()
