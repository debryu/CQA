import torch

def binary_ece_from_logits(
    logits: torch.Tensor,   # shape (N,)
    labels: torch.Tensor,   # shape (N,)
    n_bins: int = 15,
):
    probs = torch.sigmoid(logits)
    confidences = torch.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).long()
    accuracies = predictions.eq(labels).float()

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.float().mean()
        if prop > 0:
            ece += prop * torch.abs(
                accuracies[in_bin].mean() - confidences[in_bin].mean()
            )

    return ece.item()

def cbm_concept_ece(
    concept_logits: torch.Tensor,   # (N, C)
    concept_labels: torch.Tensor,   # (N, C)
    n_bins: int = 15,
):
    C = concept_logits.shape[1]
    eces = []

    for c in range(C):
        ece_c = binary_ece_from_logits(
            logits=concept_logits[:, c],
            labels=concept_labels[:, c],
            n_bins=n_bins
        )
        eces.append(ece_c)

    
    return {
        'eces': eces,
        'ece': sum(eces) / C
    }
