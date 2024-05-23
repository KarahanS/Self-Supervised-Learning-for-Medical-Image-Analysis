import torch
import torch.nn.functional as F


# only for n_views = 2 (SimCLR implementation)
def nt_xent(z, temperature):

    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)  # (512, 512)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(
        self_mask, -9e15
    )  # set cosine similarity to itself to -9e15 (nearly 0)

    # Find positive example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim /= temperature
    loss = (-cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)).mean()

    # Get ranking position of positive example
    comb_sim = torch.cat(
        [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
        # First position positive example
        dim=-1,
    )

    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    return loss, sim_argsort
