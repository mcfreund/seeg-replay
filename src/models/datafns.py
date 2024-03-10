import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
import ipdb


def chunk_data(dps, data):

    # Simple referencing
    slen       = data.slen
    nch        = dps.nch
    stride     = dps.stride
    w_len_hist = dps.w_len_hist
    w_len_pred = dps.w_len_pred

    # Initialize input, target data arrays e.g., (dim_batch, dim_input)
    nclps  = int(np.floor((slen - w_len_hist - w_len_pred) / stride) + 1)
    inputs = torch.zeros(  nclps, w_len_hist*nch)
    targs  = torch.zeros(  nclps, w_len_pred    )

    # Copy data chunks
    for t in range(0, nclps):
        ind_beg = t*stride
        ind_end = t*stride + w_len_hist
        ind_prd = t*stride + w_len_hist + w_len_pred

        # Flatten targets in column major order (electrode contiguity)
        inputs[t, :] = torch.tensor(data.X[ind_beg:ind_end,:].flatten('F'))
        targs[ t, :] = torch.tensor(data.Y[ind_end:ind_prd])

    # Return input and target data
    return inputs, targs, nclps

class DataParams:
    def __init__(self, nch = 1, w_len_hist = 5000, w_len_pred = 500, stride = 10):
        # Window lengths, stride length (history and prediction)
        self.w_len_hist = w_len_hist
        self.w_len_pred = w_len_pred
        self.stride     = stride
        self.nch        = nch
        self.nclps      = None

class DataClass:
    def __init__(self, dps):
        # Load ECOG data
        #dir = '/oscar/data/brainstorm-ws/megagroup_data/epochs/e0010GP/Encoding/'
        self.dir  = '/home/dan/projects/work/megagroup_data/epochs/e0010GP/Encoding/'
        self.subj = pd.read_csv(self.dir + 'e0010GP_Encoding_no60hz_ref_bp_clip-epo.csv', sep=',')

        # Use nch electrodes to predict 1 held out
        self.X = np.array(self.subj.iloc[:,4:(4 + dps.nch)])
        self.Y = np.array(self.subj.iloc[:,4] )

        # Series length (number of samples)
        self.slen = self.Y.shape[0]


# Define the custom dataset
class RandomVecIO(Dataset):
    """Custom Dataset for fetching [batch_size] random samples of length [len_inp] + [len_trg] from a vector."""
    def __init__(self, data_vector, len_inp, len_trg, n_samples, overlap):
        """
        Args:
            data_vector (array-like): Source vector from which to sample.
            len_inp (int): The length of the input  component of each sample.
            len_trg (int): The length of the target component of each sample.
        """
        # Save the data vector itself
        self.data_vector = data_vector

        # Save parameters
        self.len_inp = int(len_inp)
        self.len_trg = int(len_trg)
        self.len_cut = int(len_inp + len_trg - overlap)
        self.overlap = int(overlap)

        # How many samples to get?
        self.n_samples = n_samples

    def __len__(self):
        """Return the total number of possible samples."""
        return len(self.data_vector) - self.len_cut + 1

    def getbatch(self):
        """Return a batch of random samples pairs."""
        pairs = [item for pair in [self._getitem() for i in range(self.n_samples)] for item in pair]
        return torch.stack(pairs[0::2]), torch.stack(pairs[1::2])

    def _getitem(self):
        """Fetch a random sample of length len_inp+len_trg from the vector."""
        # Randomly select the start position for the sample
        idx_beg = np.random.randint(0, len(self.data_vector) - self.len_cut + 1)
        
        # Slice the sample from the vector
        sample = self.data_vector[idx_beg : idx_beg + self.len_cut]
        
        # Split the sample into input and target
        inp = sample[: self.len_inp]
        trg = sample[(self.len_inp - self.overlap):]
        
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(trg, dtype=torch.float32)


