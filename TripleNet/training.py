import torch
import torch.nn.functional as F


class RankedListLoss(torch.nn.Module):
    def __init__(self, margin=35, boundary=40, Tp=-0.5, Tn=-0.5):
        super(RankedListLoss, self).__init__()
        self.margin = margin
        self.boundary = boundary
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, distance_positive, distance_negative):
        distance_positive = torch.abs(distance_positive)
        distance_negative = torch.abs(distance_negative)

        weight_negativ = torch.exp(self.Tn * (self.boundary - distance_negative))
        weight_positiv = torch.exp(self.Tp * (distance_positive - (self.boundary - self.margin)))
        positive_pairs = F.relu(distance_positive - (self.boundary - self.margin))
        negative_pairs = F.relu(self.boundary - distance_negative)
        loss_positive = torch.sum((weight_positiv / torch.sum(weight_positiv)) * positive_pairs)
        loss_negative = torch.sum((weight_negativ / torch.sum(weight_negativ)) * negative_pairs)
        loss = torch.mean(1 / 2 * loss_positive +
                          1 / 2 * loss_negative)
        return loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def train_triplet(data_loader, model, optimizer, device, in_channels):
    loss_func_ranked_list = RankedListLoss()
    ranked_list_loss = 0

    for i, data in enumerate(data_loader, 0):
        anchor, positive, negative = data
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor = torch.reshape(anchor, (anchor.shape[0], -1, in_channels))
        positive = torch.reshape(positive, (positive.shape[0], -1, in_channels))
        negative = torch.reshape(negative, (negative.shape[0], -1, in_channels))
        optimizer.zero_grad()
        output_p, output_n = model(anchor, positive, negative)
        ranked_list_loss = loss_func_ranked_list(output_p, output_n)
        ranked_list_loss.backward()
        optimizer.step()
        ranked_list_loss += ranked_list_loss.item()
    return ranked_list_loss


def validate(data_loader, model, device):
    loss_func_ranked_list = RankedListLoss()
    ranked_list_loss = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            output_p, output_n = model(anchor, positive, negative)
            ranked_list_loss = loss_func_ranked_list(output_p, output_n)
            ranked_list_loss += ranked_list_loss.item()
    return ranked_list_loss
