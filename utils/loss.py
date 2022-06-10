import torch
from torch import nn, softmax
from torch import Tensor

__all__ = ("DiceLoss", "Accuracy")

"""
loss and metrics
    in common: loss + metrics = 1


"""
class BasicMetric(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder_(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _one_hot_encoder(self, target, input_shape):
        return self._one_hot_encoder_(target) if self.n_classes > 1 else target.reshape(*input_shape).float()
    
    def _soft_max_encoder(self, inputs):
        return torch.softmax(inputs, dim=1) if self.n_classes > 1 else torch.sigmoid(inputs)

    def process(self, inputs, target, softmax=False):
        """
        inputs: B, n_classes, H, W: \in (-inf, +inf)
        target: B, H, W: \in (0, n_classes - 1)
        ret:
            inputs: \in possibilty format: \in (0, 1) and sum at dim=1 is 1. and keep float
            target: B, n_classes, H, W: \in {0, 1} and any -> float
        """
        if softmax:
            inputs = self._soft_max_encoder(inputs)
        target = self._one_hot_encoder(target, inputs.shape)
        return inputs, target



class DiceLoss(BasicMetric):
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score) # func of dice_loss is derivable(continuous)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        input: batch, classes, ... float
        target: batch, ... int
        """
        inputs, target = self.process(inputs, target, softmax)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class Criterion(torch.nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.n_classes = num_classes
        self.loss = torch.nn.BCELoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()
    
    def forward(self, inputs, target, softmax=True):
        """
        input: batch, classes, ... float
        target: batch, ... int
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1) if self.n_classes > 1 else torch.sigmoid(inputs) # inputs.sum(-1) (== 11)
        if self.n_classes == 1:
            target = target.reshape(*inputs.shape).float()
        else:
            target = target.long()
        return self.loss(inputs, target)

class Accuracy(BasicMetric):

    def forward(self, inputs, target, weight=None, softmax=False):
        inputs, target = self.process(inputs, target, softmax)
        if weight is None:
            weight = [1] * self.n_classes
        inputs = self._one_hot_encoder(inputs.argmax(dim=1), inputs.shape) # 
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        acc_calc = 0.0
        for i in range(0, self.n_classes):
            acc = self._acc_metric(inputs[:, i], target[:, i])
            acc_calc += acc * weight[i]
        return acc_calc / self.n_classes
    
    def _acc_metric(self, inputs:Tensor, target:Tensor):
        """
        input & target's shape: B, H, W
        """
        total_sum = (target == target).sum().float()
        acc_sum = (inputs == target).sum().float()
        return acc_sum / total_sum
        
        


class Sensitivity(BasicMetric):

    def forward(self, x, y, softmax=False):
        pass

class Specificity(BasicMetric):

    def forward(self, x, y, softmax=False):
        pass

class JSI(BasicMetric):

    def forward(self, x, softmax=False):
        pass

class MCC(BasicMetric):

    def forward(self, x, softmax=False):
        pass

# deprecated: /home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/Pytorch-UNet/utils/dice_score.py/
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    print("Warning dice_coeff is deprecated method")
    pass

# deprecated /home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/Pytorch-UNet/utils/dice_score.py/
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    print("Warning multiclass_dice_coeff is deprecated method")
    pass

# deprecated /home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/Pytorch-UNet/utils/dice_score.py/
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    print("Warning dice_loss is deprecated method, use DiceLoss class instead or use Pytorch-UNet/utils/dice_score.py's dice_loss")
    pass


if __name__ == "__main__":
    print("---- start utils.loss Diceloss test ----")

    loss_1 = DiceLoss(1).to("cuda:0")
    acc_1 = Accuracy(1).to("cuda:0")
    data_x = torch.randn(16, 1, 7, 7).to("cuda:0")
    data_y = torch.randint(0, 2 ,(16, 1, 7, 7)).to("cuda:0")
    print("\tnum_classes = 1")
    print("\tdice loss value: {:1.4}".format(loss_1(data_x, data_y, softmax=True).item()))
    print("\tacc value: {:1.4}".format(acc_1(data_x, data_y, softmax=True).item()))
    
    loss_3 = DiceLoss(3).to("cuda:0")
    acc_3 = Accuracy(3).to("cuda:0")
    data_x = torch.randn(16, 3, 7, 7).to("cuda:0")
    data_y = torch.randint(0, 4 ,(16, 7, 7)).to("cuda:0")
    print("\tnum_classes = 3")
    print("\tdice loss value: {:1.4}".format(loss_3(data_x, data_y, softmax=True).item()))
    print("\tacc value: {:1.4}".format(acc_3(data_x, data_y, softmax=True).item()))

    print("---- PASS end utils.loss Diceloss test ----")