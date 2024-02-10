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
# Z,x |-> G_\theta(Z,x)
############################################################################

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import *
from utils import ReverseNoise, sample

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
models_loaded = [ReverseNoise(dim=4) for i in range(9)]
for i in range(9):
  models_loaded[i].load_state_dict(torch.load(f'parameters/model_{i}.pth', map_location=device))

def generative_model(noise, scenario):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    scenario: ndarray with shape (n_samples, n_scenarios=9)
        input categorical variable of the conditional generative model
    """
    # See below an example
    # ---------------------
    latent_variable = noise[:, :4]
    scenario_index = np.argmax(scenario[0])
    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    model = models_loaded[scenario_index]
    return sample(model, noise=latent_variable, dim=4)




