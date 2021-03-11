import torch

def get_recall_at_k(prediction, gt_truth, k):
    r"""
    :param prediction: shape (batch_size, num_class)
    :param gt_truth: shape (batch_size,1) postion of grount trunth in the list candidated classes
    :param k: top k
    :return:
    """
    batch_size = prediction.shape[0]
    values, indices = torch.sort(prediction, dim=1, descending=True)
    indices = indices[:,:k]
    gt_truth = gt_truth.unsqueeze(1).expand(batch_size,k)
    num_recall_k = torch.sum(indices==gt_truth)
    return num_recall_k