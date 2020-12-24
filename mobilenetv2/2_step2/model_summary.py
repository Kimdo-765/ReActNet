from torchsummary import summary
from reactnetv2 import reactnetv2
from cifar10_models import mobilenet_v2

model = mobilenet_v2()
print("MobileNetV2")
summary(model.cuda(),(3,32,32))

model = reactnetv2()
print("ReActNetV2")
summary(model.cuda(),(3,32,32))
