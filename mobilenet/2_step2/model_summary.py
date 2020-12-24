from torchsummary import summary
from reactnet import reactnet
from cifar10_models import MobileNet

model = MobileNet()
summary(model.cuda(),(3,32,32))

model = reactnet()
summary(model.cuda(),(3,32,32))
