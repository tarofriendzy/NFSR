
import torch


def unsorted_segment_sum(data, segment_ids, num_segments):
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().cuda()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).cuda().scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor
