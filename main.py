from embeddings.data import datasets
from embeddings.core.gcn_embeddings import get_gcn_node_embeddings

from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from keras import Input, Model, losses
from tensorflow.keras import utils

from tensorflow.keras.callbacks import EarlyStopping

from embeddings.utils import *

from GAT import gat


def autoencoder(data, encoding_dim):
    x_train, x_test = model_selection.train_test_split(data, test_size=0.3)
    input_dim = data.shape[1]

    inputs = Input(shape=(input_dim,))
    cp = closest_power2(input_dim) / 2

    #Encoder Layers
    e = inputs
    i = cp
    j = 1
    while i > encoding_dim:
        e = Dense(name="encoder" + str(j), units=i, activation='relu')(e)
        j += 1
        i /= 2
    bottleneck = Dense(name="bottleneck", units=encoding_dim, activation='relu')(e)

    #Decoder Layers
    d = bottleneck
    j = 1
    while i < cp:
        i *= 2
        d = Dense(name="decoder" + str(j), units=i, activation='relu')(d)
        j += 1
    output = Dense(name="output", units=input_dim, activation='sigmoid')(d)


    autoencoder = Model(inputs=inputs, outputs=output)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.summary()
    utils.plot_model(autoencoder, to_file='autoencoder_model.png', show_shapes=True, show_layer_names=True)

    autoencoder.fit(x_train, x_train, epochs=50 ,batch_size=32,
                    shuffle=False, validation_data=(x_test, x_test))

    #Dimension reduction
    encoder = Model(inputs=inputs, outputs=bottleneck)
    encoded = encoder.predict(data)
    return encoded


def DNN(g,data):
    node_subjects = g.node_subjects
    n_labels = g.get_number_of_subjects()

    x_train, x_val, x_test, y_train, y_val, y_test = getxy(g, data, node_subjects, 1300, 1000)


    n_features = data.shape[1]
    inputs = Input(name="input", shape=(n_features,))

    h1 = Dense(name="h1", units=int(round((n_features+1)/2)), activation='relu')(inputs)
    h1 = Dropout(name="drop1", rate=0.2)(h1)

    h2 = Dense(name="h2", units=int(round((n_features + 1) / 4)), activation='relu')(h1)
    h2 = Dropout(name="drop2", rate=0.2)(h2)

    outputs = Dense(name="output", units=n_labels, activation='softmax')(h2)

    model = Model(inputs=inputs, outputs=outputs, name="DeepNN")
    utils.plot_model(model, to_file='DNN_model.png', show_shapes=True, show_layer_names=True)
    #model.summary()
    #font = ImageFont.truetype("arial.ttf", 12)
    #visualkeras.layered_view(model, legend=True, font=font).show()

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val),
              shuffle=True, callbacks=[es_callback], verbose=0)

    y_pred = model.predict(x_test)
    evaluation(y_test, y_pred)


def cora():
    dataset = datasets.Cora()
    g = dataset.load()

    features = g.features.to_numpy()

    gcn_embeddings = np.load('Cora_GCN_embeddings.npy').squeeze(0)
    word_autoencoder16_embeddings = np.load('Cora_16_autoencoder.npy')
    print(gcn_embeddings.shape)

    #emb = get_gcn_node_embeddings(g)
    #np.save('Cora_GCN_embeddings.npy', emb)

    #emb = autoencoder(features, 16)
    #np.save('Cora_16_autoencoder.npy', emb)

    #concatenated_embeddings = np.concatenate((gcn_embeddings, word_autoencoder16_embeddings), axis=1)
    #plused_embeddings = np.add(gcn_embeddings, word_autoencoder16_embeddings)

    #con2 = np.concatenate((gcn_embeddings, features), axis=1)
    #emb = autoencoder(con2, 32)
    #np.save('Cora_con2.npy', emb)

    con2 = np.load('Cora_autoencoder_Concatenated.npy')


    DNN(g, gcn_embeddings)


def citeseer():
    dataset = datasets.CiteSeer()
    g = dataset.load()

    features = g.features.to_numpy()

    gcn_embeddings = np.load('CiteSeer_GCN_embeddings.npy').squeeze(0)
    word_autoencoder16_embeddings = np.load('CiteSeer_16_autoencoder.npy')

    #emb = get_gcn_node_embeddings(g)
    #np.save('CiteSeer_GCN_embeddings.npy', emb)

    #emb = autoencoder(features, 16)
    #np.save('CiteSeer_16_autoencoder.npy', emb)

    concatenated_embeddings = np.concatenate((gcn_embeddings, word_autoencoder16_embeddings), axis=1)
    plused_embeddings = np.add(gcn_embeddings, word_autoencoder16_embeddings)

    DNN(g, plused_embeddings)

def pubmed():
    dataset = datasets.PubMedDiabetes()
    g = dataset.load()
    node_subjects = g.node_subjects

    features = g.features.to_numpy()

    gcn_embeddings = np.load('PubMed_GCN_embeddings.npy').squeeze(0)
    word_autoencoder16_embeddings = np.load('PubMed_16_autoencoder.npy')

    #emb = get_gcn_node_embeddings(g)
    #np.save('PubMed_GCN_embeddings.npy', emb)

    #emb = autoencoder(features, 16)
    #np.save('PubMed_16_autoencoder.npy', emb)

    concatenated_embeddings = np.concatenate((gcn_embeddings, word_autoencoder16_embeddings), axis=1)
    plused_embeddings = np.add(gcn_embeddings, word_autoencoder16_embeddings)

    DNN(g, gcn_embeddings)




def cora_gat2():
    dataset = datasets.Cora()
    g = dataset.load()

    gat_embeddings = gat.train(dataset=dataset, gatv2=False)
    #np.save('Cora_GAT_embeddings.npy', gat_embeddings)

    gatv2_embeddings = gat.train(dataset=dataset, gatv2=True)
    #np.save('Cora_GATV2_embeddings.npy', gatv2_embeddings)
    mkmkmkm

    gat_embeddings = np.load('Cora_GAT_embeddings.npy')
    gatv2_embeddings = np.load('Cora_GATV2_embeddings.npy')


    features = g.features.to_numpy()
    #emb = autoencoder(features, 16)
    #np.save('Cora_16_autoencoder.npy', emb)

    word_autoencoder16_embeddings = np.load('Cora_16_autoencoder.npy')

    gat_concatenated_embeddings = np.concatenate((gat_embeddings, word_autoencoder16_embeddings), axis=1)
    gatv2_concatenated_embeddings = np.concatenate((gatv2_embeddings, word_autoencoder16_embeddings), axis=1)
    gat_plused_embeddings = np.add(gat_embeddings, word_autoencoder16_embeddings)
    gatv2_plused_embeddings = np.add(gatv2_embeddings, word_autoencoder16_embeddings)

    print("====================================")
    #DNN(g,gat_embeddings)
    #DNN(g,gatv2_embeddings)
    print("====================================")

    DNN(g, gat_plused_embeddings)
    DNN(g, gatv2_plused_embeddings)
    print("====================================")

    DNN(g, gat_concatenated_embeddings)
    DNN(g, gatv2_concatenated_embeddings)
    print("====================================")


def pubmed_gat2():
    dataset = datasets.PubMedDiabetes()
    g = dataset.load()

    #gat_embeddings = gat.train(dataset=dataset, gatv2=False)
    #np.save('PubMed_GAT_embeddings.npy', gat_embeddings)

    #gatv2_embeddings = gat.train(dataset=dataset, gatv2=True)
    #np.save('PubMed_GATV2_embeddings.npy', gatv2_embeddings)

    #gat_embeddings = np.load('PubMed_GAT_embeddings.npy')
    gatv2_embeddings = np.load('PubMed_GATV2_embeddings.npy')

    word_autoencoder16_embeddings = np.load('PubMed_16_autoencoder.npy')

    #gat_concatenated_embeddings = np.concatenate((gat_embeddings, word_autoencoder16_embeddings), axis=1)
    gatv2_concatenated_embeddings = np.concatenate((gatv2_embeddings, word_autoencoder16_embeddings), axis=1)
    #gat_plused_embeddings = np.add(gat_embeddings, word_autoencoder16_embeddings)
    gatv2_plused_embeddings = np.add(gatv2_embeddings, word_autoencoder16_embeddings)

    print("====================================")
    #DNN(g,gat_embeddings)
    DNN(g,gatv2_embeddings)
    print("====================================")

    #DNN(g, gat_plused_embeddings)
    DNN(g, gatv2_plused_embeddings)
    print("====================================")

    #DNN(g, gat_concatenated_embeddings)
    DNN(g, gatv2_concatenated_embeddings)
    print("====================================")



def citeseer_gat2():
    dataset = datasets.CiteSeer()
    g = dataset.load()

    #gat_embeddings = gat.train(dataset=dataset, gatv2=False)
    #np.save('Citeseer_GAT_embeddings.npy', gat_embeddings)

    #gatv2_embeddings = gat.train(dataset=dataset, gatv2=True)
    #np.save('Citeseer_GATV2_embeddings.npy', gatv2_embeddings)

    gat_embeddings = np.load('Citeseer_GAT_embeddings.npy')
    gatv2_embeddings = np.load('Citeseer_GATV2_embeddings.npy')

    word_autoencoder16_embeddings = np.load('CiteSeer_16_autoencoder.npy')

    gat_concatenated_embeddings = np.concatenate((gat_embeddings, word_autoencoder16_embeddings), axis=1)
    gatv2_concatenated_embeddings = np.concatenate((gatv2_embeddings, word_autoencoder16_embeddings), axis=1)
    gat_plused_embeddings = np.add(gat_embeddings, word_autoencoder16_embeddings)
    gatv2_plused_embeddings = np.add(gatv2_embeddings, word_autoencoder16_embeddings)

    print("====================================")
    DNN(g,gat_embeddings)
    DNN(g,gatv2_embeddings)
    print("====================================")

    DNN(g, gat_plused_embeddings)
    DNN(g, gatv2_plused_embeddings)
    print("====================================")

    DNN(g, gat_concatenated_embeddings)
    DNN(g, gatv2_concatenated_embeddings)
    print("====================================")



if __name__ == '__main__':
    cora_gat2()































