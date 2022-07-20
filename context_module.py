import torch
from torchvision.models import resnet50

# Simple

# model = resnet50(False)


# out = model(im)
# print(out)

# Full definition

class ContextModel(torch.nn.Module):
    def __init__(self):
        super(ContextModel, self).__init__()        
        self.model = resnet50(False)

    def forward(self, x, batch_size, num_context):
        x = self._flatten(x, batch_size, num_context)
        x = self.model(x)
        x = self._unflatten(x, batch_size, num_context)
        return x
    
    def _flatten(self, x, batch_size, num_context):
        bs, nc, channels, len1, len2 = x.shape
        x = torch.reshape(x, (batch_size*num_context, channels, len1, len2))
        return x
    
    def _unflatten(self, x, batch_size, num_context):
        flattened_batch_size, vector_len = x.shape
        x = torch.reshape(x, (batch_size, num_context, vector_len))
        return x

im = torch.rand((16,3,3,224,224))
cm = ContextModel()
out = cm(im, 16, 3)
print(out.shape)