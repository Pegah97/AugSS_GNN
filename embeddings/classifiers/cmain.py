
import umap as umap

from embeddings.data import datasets
from embeddings.core.gcn_embeddings import get_gcn_node_embeddings
import numpy as np

import keras
from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from keras import Input, Model, optimizers, losses, backend as K
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import visualkeras
from PIL import ImageFont
import seaborn as sns

from embeddings.core.graph import MyGraph

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import f1_score , accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn import preprocessing, model_selection
from embeddings .utils import *

def new_classification(data):
    node_subjects = g.node_subjects
    n_labels = g.get_number_of_subjects()

    x_train, x_val, x_test, y_train, y_val, y_test = getxy(g, data, node_subjects, 600, 700)

    n_features = data.shape[1]
    inputs = Input(name="input", shape=(n_features,))

    h1 = Dense(name="h1", units=int(round((n_features + 1) / 2)), activation='relu')(inputs)
    h1 = Dropout(name="drop1", rate=0.2)(h1)

    h2 = Dense(name="h2", units=int(round((n_features + 1) / 4)), activation='relu')(h1)
    h2 = Dropout(name="drop2", rate=0.2)(h2)

    outputs = Dense(name="output", units=n_labels, activation='softmax')(h2)

    model = Model(inputs=inputs, outputs=outputs, name="DeepNN")
    # font = ImageFont.truetype("arial.ttf", 12)
    # visualkeras.layered_view(model, legend=True, font=font).show()

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), shuffle=True)

    # Evaluate model
    test_metrics = model.evaluate(x=x_test, y=y_test)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    '''
    pred_train = model.predict(x_train)
    scores = model.evaluate(x_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(x_test)
    scores2 = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))
    #auc, pr, f_score = evaluate_preds(y_test.ravel(), pred_test.ravel())
    '''


def MLP(g, data, node_subjects):

    x_train, x_val, x_test, y_train, y_val, y_test = getxy(g, data, node_subjects, 600, 700)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
    #s = clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    macro_f1_score(y_test,y_pred)


def macro_f1_score(y_true, y_pred):
    macroF1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print("macroF1: " + str(macroF1))
    print("accuracy: " + str(accuracy))
    return macroF1


if __name__ == '__main__':
    np_path = 'C:/Users/mehdi/PycharmProjects/graph_embeddings_pro'
    dataset = datasets.Cora()
    g = dataset.load()
    node_subjects = g.node_subjects

    graph = g
    data = g.features.to_numpy()
    subjects = node_subjects

    #data = np.load(np_path + '/Cora_GCN_embeddings.npy').squeeze(0)

    MLP(graph, data, subjects)







