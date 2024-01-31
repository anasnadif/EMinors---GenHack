#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z |-> G_\theta(Z)
############################################################################
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



path = "https://raw.githubusercontent.com/generative-hackathon/Genhack3/master/data/"
station_1 = pd.read_csv(path+"station_49.csv")
station_2 = pd.read_csv(path+"station_80.csv")
station_3 = pd.read_csv(path+"station_40.csv")
station_4 = pd.read_csv(path+"station_63.csv")

Q = np.array([3.3241, 5.1292, 6.4897, 7.1301])
cond1 = station_1['W_13'] + station_1['W_14'] + station_1['W_15']  < Q[0]
cond2 = station_2['W_13'] + station_2['W_14'] + station_2['W_15'] < Q[1]
cond3 = station_3['W_13'] + station_3['W_14'] + station_3['W_15']  < Q[2]
cond4 = station_4['W_13'] + station_4['W_14'] + station_4['W_15'] < Q[3]
condTotal = cond1*cond2*cond3*cond4

Y_1 = station_1[condTotal][["YIELD"]].values
Y_2 = station_2[condTotal][["YIELD"]].values
Y_3 = station_3[condTotal][["YIELD"]].values
Y_4 = station_4[condTotal][["YIELD"]].values

extreme_yields = {'Y1': list(Y_1.reshape(-1)),
                  'Y2': list(Y_2.reshape(-1)), # to normalize the distribution
                  'Y3': list(Y_3.reshape(-1)),
                  'Y4': list(Y_4.reshape(-1))}

extreme_yields_df = pd.DataFrame(extreme_yields)

scaler = MinMaxScaler(feature_range=(-1, 1))
extreme_yields = scaler.fit_transform(extreme_yields_df)



def make_generator_model(latent_dim=16):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    model.add(tf.keras.layers.Dense(4, activation=('tanh')))
    return model




# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_variable = noise[:, 50]  # choose the appropriate latent dimension of your model
    generator = make_generator_model(50)
    generator.load_weights('parameters/generator_weights_0.0891.h5')
    generated_yields = generator(latent_variable, training=False)
    generated_yields_unscaled = scaler.inverse_transform(generated_yields)
    return generated_yields_unscaled # G(Z)




