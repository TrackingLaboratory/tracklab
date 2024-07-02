import torch


def identity(tracks, dets):
    return tracks, dets


def sum(tracks, dets):
    """
    Merge the tokens of tracks and dets by summing them.
    """
    tracks.tokens = torch.stack(list(tracks.tokens.values()), dim=3).sum(dim=3)
    dets.tokens = torch.stack(list(dets.tokens.values()), dim=3).sum(dim=3)
    return tracks, dets


def concat(tracks, dets):
    """
    Merge the tokens of tracks and dets by concatenating them.
    """
    tracks.tokens = torch.concat(list(tracks.tokens.values()), dim=-1)
    dets.tokens = torch.concat(list(dets.tokens.values()), dim=-1)
    return tracks, dets


merge_token_strats = {
    "sum": sum,
    "identity": identity,
    "concat": concat,
}