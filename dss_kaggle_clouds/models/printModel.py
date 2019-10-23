from torchvision import models
from torchsummary import summary
import fcn8s
net = fcn8s.FCN8s(num_classes=4).cuda()
# vgg = models.vgg16()
summary(net, (3, 224, 224))

