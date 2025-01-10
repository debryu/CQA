from metrics.dci_framework import _compute_dci

def DCI_wrapper(representation_train,concept_train,representation_val,concept_val, level = 'INFO'):
  '''
  representation_train: torch.Tensor of shape (n_features, n_samples)
  concept_train: torch.Tensor of shape (n_features, n_samples)
  representation_val: torch.Tensor of shape (n_features, n_samples)
  concept_val: torch.Tensor of shape (n_features, n_samples)
  '''

  dci = _compute_dci(representation_train.T,concept_train.T,representation_val.T,concept_val.T)
  return dci
