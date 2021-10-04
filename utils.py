import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def overlaySegment(gray1, seg1, colors, flag=False):
    H, W = seg1.squeeze().size()
    segs1 = F.one_hot(seg1.long(), 3).float().permute(2, 0, 1)[:3]
    seg_color = torch.mm(segs1.view(3, -1).t().cpu(), colors[:3, :]).view(H, W, 3)
    alpha = torch.clamp(1.0 - 0.5 * (seg1 > 0).float(), 0, 1.0)

    overlay = (gray1.cpu() * alpha.cpu()).unsqueeze(2) + seg_color.cpu() * (1.0 - alpha.cpu()).unsqueeze(2)
    if flag:
        plt.imshow(overlay.numpy());
        plt.axis('off');
        plt.show()
    return overlay, seg_color.cpu()


def save_model(model, path):
    torch.save(model.state_dict(), path+'.pt')
    torch.save(model, path + '_model.pt')


def load_model(path):
    return torch.load(path + '_model.pt'), torch.load(path + '.pt')


def iou_coef(y_true, y_pred, smooth=1):
    axis = tuple([2, 3, 4])
    intersection = torch.sum(torch.abs(y_true * y_pred), axis)
    union = torch.sum(y_true, axis) + torch.sum(y_pred, axis) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), 0)
    return iou

def iou(pred, target, n_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    #y_true = y_true[:,1:]
    #y_pred = y_pred[:,1:]
    axes = tuple(range(1, len(y_pred.shape) - 1))
    axes = tuple([2, 3, 4])
    numerator = 2 * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)
    print((numerator + epsilon) / (denominator + epsilon))
    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))


# Declare the Dice Loss
def torch_dice_coef_loss(y_true, y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))
