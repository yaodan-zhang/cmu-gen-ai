import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
#torch.save(model.state_dict(), 'model_weights.pth')

torch.save(model, 'model.pth')


#model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
#model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

model = torch.load('model.pth', weights_only=False),

#print(model.eval())