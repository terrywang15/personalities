import random
import time
import os
import pandas as pd
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def load_data(fp, test_prop):
    """
    loads and transforms data for modeling
    takes a fp to a csv file
    returns a pandas dataframe
    """

    # load data
    data = pd.read_csv(fp)

    # split train and test
    # how many rows to sample
    n_samps = int(data.shape[0] * (1 - test_prop))
    # sample from row indices
    train_idx = random.sample(range(data.shape[0]), n_samps)
    test_idx = list(set(range(data.shape[0])) - set(train_idx))

    return data.loc[train_idx,:], data.loc[test_idx,:]


def create_ae(input_dim, layer_dim, encoder_dim, dropout_rate, activation_func):

    # Encoder
    input_encoder = Input(shape=(input_dim, ))
    encoder1 = Dense(layer_dim, activation=activation_func)(input_encoder)
    e_bnorm1 = BatchNormalization(axis=1)(encoder1)
    e_dropout1 = Dropout(dropout_rate)(e_bnorm1)
    encoder2 = Dense(layer_dim, activation=activation_func)(e_dropout1)
    e_bnorm2 = BatchNormalization(axis=1)(encoder2)
    e_dropout2 = Dropout(dropout_rate)(e_bnorm2)
    encoder3 = Dense(layer_dim, activation=activation_func)(e_dropout2)
    e_bnorm3 = BatchNormalization(axis=1)(encoder3)
    e_dropout3 = Dropout(dropout_rate)(e_bnorm3)
    output_encoder = Dense(encoder_dim, activation="sigmoid")(e_dropout3)

    # Decoder
    input_decoder = Input(shape=(encoder_dim, ))
    decoder1 = Dense(layer_dim, activation=activation_func)(input_decoder)
    d_bnorm1 = BatchNormalization(axis=1)(decoder1)
    d_dropout1 = Dropout(dropout_rate)(d_bnorm1)
    decoder2 = Dense(layer_dim, activation=activation_func)(d_dropout1)
    d_bnorm2 = BatchNormalization(axis=1)(decoder2)
    d_dropout2 = Dropout(dropout_rate)(d_bnorm2)
    decoder3 = Dense(layer_dim, activation=activation_func)(d_dropout2)
    d_bnorm3 = BatchNormalization(axis=1)(decoder3)
    d_dropout3 = Dropout(dropout_rate)(d_bnorm3)
    output_decoder = Dense(input_dim, activation=activation_func)(d_dropout3)

    encoder = Model(inputs=input_encoder, outputs=output_encoder, name='encoder')
    decoder = Model(inputs=input_decoder, outputs=output_decoder)
    autoencoder = Model(inputs=input_encoder, outputs=decoder(encoder(input_encoder)), name='ae')

    return encoder, decoder, autoencoder


# load data
X_train, X_test = load_data("data/data_complete_only.csv", 0.3)

input_dim = X_train.shape[1]
layer_dim = 128
encoder_dim = 4
dropout_rate = 0.1
learning_rate = 0.001
activ_func = 'tanh'

# Make timestamp
ts_str = "models/" + time.strftime("%Y-%m-%d %H-%M", time.gmtime())
# other parameters
n_epochs = 1000
n_batch = 600
# create directory if not exist
if not os.path.exists(ts_str):
    os.makedirs(ts_str)


encoder_model, decoder_model, ae_model = create_ae(input_dim, layer_dim, encoder_dim, dropout_rate, activ_func)

ae_model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')

n_epochs = 50
batch_size = 150
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

ae_model.fit(X_train, X_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test),
        callbacks=[earlystopping])

# encoder_out.save('models/encoder_out_model.h5')
encoder_model.save(ts_str + '/encoder_model.h5')
ae_model.save(ts_str + '/ae_model.h5')
