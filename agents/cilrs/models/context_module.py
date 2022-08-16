import torch
from torchvision.models import resnet50

# Simple

# model = resnet50(False)


# out = model(im)
# print(out)

# Full definition

class ContextModel(torch.nn.Module):
    def __init__(self, model):
        super(ContextModel, self).__init__()        
        self.model = model

    def forward(self, x, batch_size, num_context):
        x = self._flatten(x, batch_size, num_context)
        x = self.model(x)
        x = self._unflatten(x, batch_size, num_context)
        x = self._aggregate(x)
        return x
    
    def _flatten(self, x, batch_size, num_context):
        bs, nc, channels, len1, len2 = x.shape
        x = torch.reshape(x, (batch_size*num_context, channels, len1, len2))
        return x
    
    def _unflatten(self, x, batch_size, num_context):
        flattened_batch_size, vector_len = x.shape
        x = torch.reshape(x, (batch_size, num_context, vector_len))
        return x

    def _aggregate(self, x):
        # Calculates mean over context dimension
        return torch.mean(x, dim=1)

# im = torch.rand((16,3,3,224,224))
# cm = ContextModel()
# out = cm(im, 16, 3)
# print(out.shape)


class ContextModule(torch.nn.Module):
    def __init__(self, perception_model, measurement_model):
        super(ContextModule, self).__init__()        
        self.p_model = perception_model
        self.m_model = measurement_model

    def forward(self, x_im, x_m, batch_size=1, num_context=3):       
        x_im, x_m = self._flatten(x_im, x_m, batch_size, num_context)
        x_im, x_m = self.p_model(x_im), self.m_model(x_m)
        x_im, x_m = self._unflatten(x_im, x_m, batch_size, num_context)
        x_im, x_m = self._aggregate(x_im), self._aggregate(x_m)
        return x_im, x_m

    def _flatten(self, x_im, x_m, batch_size, num_context):
        bs, nc, channels, len1, len2 = x_im.shape
        x_im = torch.reshape(x_im, (batch_size*num_context, channels, len1, len2))

        bs, nc, vector_len = x_m.shape
        x_m = torch.reshape(x_m, (batch_size*num_context, vector_len))
        return x_im, x_m
    
    def _unflatten(self, x_im, x_m, batch_size, num_context):
        flattened_batch_size, vector_len = x_im.shape
        x_im = torch.reshape(x_im, (batch_size, num_context, vector_len))
        
        flattened_batch_size, vector_len = x_m.shape
        x_m = torch.reshape(x_m, (batch_size, num_context, vector_len))
        return x_im, x_m

    def _aggregate(self, x):
        # Calculates mean over context dimension
        return torch.mean(x, dim=1)

class ContextPerceptionModule(torch.nn.Module):
    def __init__(self, perception_model):
        super(ContextModel, self).__init__()        
        self.model = perception_model

    def forward(self, x, batch_size, num_context):# batch_size and num_context to outter model also
        x = self._flatten(x, batch_size, num_context)
        x = self.model(x)
        x = self._unflatten(x, batch_size, num_context)
        x = self._aggregate(x)
        return x
    
    def _flatten(self, x, batch_size, num_context):
        bs, nc, channels, len1, len2 = x.shape
        x = torch.reshape(x, (batch_size*num_context, channels, len1, len2))
        return x
    
    def _unflatten(self, x, batch_size, num_context):
        flattened_batch_size, vector_len = x.shape
        x = torch.reshape(x, (batch_size, num_context, vector_len))
        return x

    def _aggregate(self, x):
        # Calculates mean over context dimension
        return torch.mean(x, dim=1)

class ContextMeasurementModule(torch.nn.Module):
    def __init__(self, full_connected_model):
        super(ContextModel, self).__init__()        
        self.model = full_connected_model

    def forward(self, x, batch_size, num_context):# batch_size and num_context to outter model also
        x = self._flatten(x, batch_size, num_context)
        x = self.model(x)
        x = self._unflatten(x, batch_size, num_context)
        x = self._aggregate(x)
        return x
    
    def _flatten(self, x, batch_size, num_context):
        bs, nc, vector_len = x.shape
        x = torch.reshape(x, (batch_size*num_context, vector_len))
        return x
    
    def _unflatten(self, x, batch_size, num_context):
        flattened_batch_size, vector_len = x.shape
        x = torch.reshape(x, (batch_size, num_context, vector_len))
        return x
            
    def _aggregate(self, x):
        # Calculates mean over context dimension
        return torch.mean(x, dim=1)

# Testing:
if __name__=="__main__":
    from networks import resnet
    from networks.fc import FC
    import torch.nn as nn

    percept = resnet.get_model('resnet34', [3,1,900,256], num_classes=1000, pretrained=False)
    fc = FC(params={'neurons': [1+4] + [128, 128],
                                       'dropouts': [0.0,0.0],
                                       'end_layer': nn.ReLU})

    cm = ContextModule(percept, fc)

    im = torch.rand((16,3,3,900,256))
    ms = torch.rand((16,3,5))

    out = cm.forward(im,ms, batch_size=16, num_context=3)
    print(out[0].shape)
    print(out[1].shape)
