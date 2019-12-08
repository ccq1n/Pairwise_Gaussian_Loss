import torch

def euclidean_dist_all(mat_ab):
    dim_v = mat_ab.size(0)
    odd_list = [i for i in range(1,dim_v,2)]
    even_list = [i for i in range(0,dim_v,2)]
    mat_a = mat_ab[odd_list, :]
    mat_b = mat_ab[even_list,:]
    dist_ab_eula = torch.sqrt(torch.sum(torch.pow((mat_a-mat_b), 2), dim=1, keepdim=True))
    return dist_ab_eula

def pairwise_gaussian_loss(dist_mat, labels_raw, n_classes, beta=0.05):
    labels_raw = labels_raw.view([-1, 1])
    one_hot = torch.zeros(labels_raw.shape[0], n_classes).scatter_(1, labels_raw.data.cpu(), 1)
    dim_v = labels_raw.size(0)
    odd_list = [i for i in range(1, dim_v, 2)]
    even_list = [i for i in range(0, dim_v, 2)]
    labels_1 = one_hot[odd_list, :]
    labels_2 = one_hot[even_list, :]
    labels_ip = torch.max(labels_1*labels_2, dim=1, keepdim=True)[0].cuda()
    dist_mat_sq = beta * (torch.pow(dist_mat, 2))
    loss = dist_mat_sq + (labels_ip-1.0)*(torch.log(torch.exp(dist_mat_sq)) - 1.0)
    loss = torch.mean(loss)
    return loss

def pairwise_sigmoid_loss(dist_mat, labels_raw, n_classes, aerfa=1):
    labels_raw = labels_raw.view([-1, 1])
    one_hot = torch.zeros(labels_raw.shape[0], n_classes).scatter_(1, labels_raw.data.cpu(), 1)
    dim_v = labels_raw.size(0)
    odd_list = [i for i in range(1, dim_v, 2)]
    even_list = [i for i in range(0, dim_v, 2)]
    labels_1 = one_hot[odd_list, :]
    labels_2 = one_hot[even_list, :]
    labels_ip = torch.max(labels_1 * labels_2, dim=1, keepdim=True)[0].cuda()
    x_b = aerfa * (dist_mat - 5.0)
    loss_sigmoid = (labels_ip - 1.0)*x_b + (torch.log(torch.exp(x_b) + 1.0))
    loss = torch.mean(loss_sigmoid)
    return loss

def pairwise_hinge_loss(dist_mat, labels_raw, n_classes, param_m=2):
    labels_raw = labels_raw.view([-1, 1])
    one_hot = torch.zeros(labels_raw.shape[0], n_classes).scatter_(1, labels_raw.data.cpu(), 1)
    dim_v = labels_raw.size(0)
    odd_list = [i for i in range(1, dim_v, 2)]
    even_list = [i for i in range(0, dim_v, 2)]
    labels_1 = one_hot[odd_list, :]
    labels_2 = one_hot[even_list, :]
    labels_ip = torch.max(labels_1 * labels_2, dim=1, keepdim=True)[0].cuda()
    max_dist = 0.5 * torch.max(torch.zeros(1).cuda(), (param_m - dist_mat))
    loss_hinge = labels_ip * (0.5 * dist_mat) + (1 - labels_ip) * max_dist
    loss = torch.mean(loss_hinge)
    return loss

def pairwise_cauchy_loss(dist_mat, labels_raw, n_classes, gamma=5):
    labels_raw = labels_raw.view([-1, 1])
    one_hot = torch.zeros(labels_raw.shape[0], n_classes).scatter_(1, labels_raw.data.cpu(), 1)
    dim_v = labels_raw.size(0)
    odd_list = [i for i in range(1, dim_v, 2)]
    even_list = [i for i in range(0, dim_v, 2)]
    labels_1 = one_hot[odd_list, :]
    labels_2 = one_hot[even_list, :]
    labels_ip = torch.max(labels_1 * labels_2, dim=1, keepdim=True)[0].cuda()
    loss_cauchy = (labels_ip * torch.log(dist_mat * (1.0/gamma))) + torch.log(1.0 + gamma/dist_mat)
    loss = torch.mean(loss_cauchy)
    return loss