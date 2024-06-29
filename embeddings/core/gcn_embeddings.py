
from __future__ import print_function
import numpy as np
import scipy.sparse as sp
from keras.layers import Input, Dropout
from tensorflow.keras import layers, optimizers, losses, Model, utils
from tensorflow.keras.callbacks import EarlyStopping

from embeddings.layers.gcn_layer import GCNLayer
from embeddings.layers.gathering_layer import GatherIndices

from matplotlib import pyplot as plt

from embeddings.utils import split_data, convert_to_numeric_arrays, evaluate_preds, evaluation

def normalize_adjacency_matrix(a):
    # Add self-connections
    a_t = a + sp.diags(np.ones(a.shape[0]) - a.diagonal())
    # Degree matrix to the power of -1/2
    d_t = sp.diags(np.power(np.array(a.sum(1)), -0.5).flatten(), 0)
    # Normalise the Adjacency matrix
    a_norm = a_t.dot(d_t).transpose().dot(d_t).tocsr().astype(dtype=np.float32)
    return a_norm


def normalize_feature_matrix(X):
    X /= X.sum(1).reshape(-1, 1)
    return X


def get_gcn_node_embeddings(g):
    ############# Paramsss
    dropout = 0.5
    n_gcn = 2
    out_dim = 16
    activation = 'relu'

    n_labels = g.get_number_of_subjects()
    node_subjects = g.get_node_subjects()

    train_subjects, val_subjects, test_subjects = split_data(node_subjects, 800, 900)
    train_targets, val_targets, test_targets = convert_to_numeric_arrays(train_subjects, val_subjects, test_subjects)

    train_indices = g.nodes_to_loc(train_subjects.index)
    val_indices = g.nodes_to_loc(val_subjects.index)
    test_indices = g.nodes_to_loc(test_subjects.index)
    all_indices = g.nodes_to_loc(node_subjects.index)

    # Normalize A,X
    adj = g.get_adjacency_matrix()
    adj = normalize_adjacency_matrix(adj)

    features_matrix = g.get_features_dense_sparse()
    features_matrix = normalize_feature_matrix(features_matrix)


    #Expand dimensions
    adj = adj.todense()
    features_input = np.expand_dims(features_matrix, 0)
    adj_input = np.expand_dims(adj, 0)

    y_train = np.expand_dims(train_targets, 0)
    y_val = np.expand_dims(val_targets, 0)
    y_test = np.expand_dims(test_targets, 0)

    x_train = [features_input, train_indices, adj_input]
    x_val = [features_input, val_indices, adj_input]
    x_test = [features_input, test_indices, adj_input]
    x_all = [features_input, all_indices, adj_input]


    x_features = Input(name="Features Input", batch_shape=(1, features_input.shape[1], features_input.shape[2]))
    x_adjacency = Input(name="Adjacency Input", batch_shape=(1, adj_input.shape[1], adj_input.shape[2]))
    x_indicies = Input(name="Indicies Input", batch_shape=(1,None), dtype="int32")


    inputs = [x_features, x_indicies, x_adjacency]

    h = Dropout(dropout)(x_features)

    #Convolution layers
    n_dim = n_gcn * out_dim
    for i in range(1, n_gcn+1):
        h = GCNLayer(name="GraphConvolution" + str(i), output_dim=int(n_dim / i), activation=activation)([h, x_adjacency])
        if i != n_gcn:
            h = Dropout(dropout)(h)


    #int(n_dim / i)

    #Gather indicies
    h = GatherIndices()([h, x_indicies])
    gcn_output = h

    #output
    output = layers.Dense(n_labels, activation='softmax')(h)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss=losses.categorical_crossentropy,
                  metrics=["acc"]
    )

    utils.plot_model(model, to_file='GCN_model.png', show_shapes=True, show_layer_names=True)


    es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=200,
        validation_data=(x_val, y_val),
        verbose=1,
        shuffle=False,
        callbacks=[es_callback],
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

    y_pred = model.predict(x_test)
    evaluation(y_test.squeeze(0), y_pred.squeeze(0))

    model_embedding = Model(inputs=inputs, outputs= gcn_output)
    embeddings = model_embedding.predict(x_all)

    return embeddings














