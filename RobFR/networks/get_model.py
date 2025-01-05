import torch
from RobFR.networks.MobileFace import MobileFace
from RobFR.networks.Mobilenet import Mobilenet
from RobFR.networks.MobilenetV2 import MobilenetV2
from RobFR.networks.ResNet import resnet
from RobFR.networks.ShuffleNetV2 import ShuffleNetV2
from RobFR.networks.ShuffleNet import ShuffleNetV1
from RobFR.networks.CosFace import CosFace
from RobFR.networks.SphereFace import SphereFace
from RobFR.networks.FaceNet import FaceNet
from RobFR.networks.ArcFace import ArcFace
from RobFR.networks.IR import IR


def getmodel(face_model, **kwargs):
    """
    Select the face model according to its name.
    :param face_model: string, the name of the face model to use.
    :param kwargs: additional arguments for model initialization.
    :return: the initialized model and its expected image shape.
    """
    device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kwargs['device'] = device

    # Default image shape
    img_shape = (112, 112)

    # Mapping of model names to constructors
    model_mapping = {
        'MobileFace': MobileFace,
        'Mobilenet': Mobilenet,
        'Mobilenet-stride1': lambda **kw: Mobilenet(stride=1, **kw),
        'MobilenetV2': MobilenetV2,
        'MobilenetV2-stride1': lambda **kw: MobilenetV2(stride=1, **kw),
        'ResNet50': lambda **kw: resnet(depth=50, **kw),
        'ResNet50-casia': lambda **kw: resnet(depth=50, dataset='casia', **kw),
        'ShuffleNet_V1_GDConv': lambda **kw: ShuffleNetV1(pooling='GDConv', **kw),
        'ShuffleNet_V2_GDConv-stride1': lambda **kw: ShuffleNetV2(stride=1, pooling='GDConv', **kw),
        'CosFace': lambda **kw: _update_img_shape(CosFace(**kw), (112, 96)),
        'SphereFace': lambda **kw: _update_img_shape(SphereFace(**kw), (112, 96)),
        'FaceNet-VGGFace2': lambda **kw: _update_img_shape(FaceNet(dataset='vggface2', use_prewhiten=False, **kw), (160, 160)),
        'FaceNet-casia': lambda **kw: _update_img_shape(FaceNet(dataset='casia-webface', use_prewhiten=False, **kw), (160, 160)),
        'ArcFace': ArcFace,
        **_add_ir_models()
    }

    if face_model not in model_mapping:
        raise NotImplementedError(f"Model '{face_model}' is not implemented.")

    try:
        model = model_mapping[face_model](**kwargs)
        model = model.to(device)
        return model, img_shape
    except Exception as e:
        raise RuntimeError(f"Error initializing model '{face_model}': {e}")


def _update_img_shape(model, shape):
    """
    Updates the global image shape and returns the model.
    :param model: the model instance.
    :param shape: tuple, the new image shape.
    :return: the model instance.
    """
    global img_shape
    img_shape = shape
    return model


def _add_ir_models():
    """
    Adds IR-based models to the model mapping.
    :return: a dictionary of IR models.
    """
    ir_models = {
        'IR50-Softmax': lambda **kw: IR(loss='Softmax', **kw),
        'IR50-Softmax-BR': lambda **kw: IR(loss='Softmax', transform='BitReduction', **kw),
        'IR50-Softmax-RP': lambda **kw: IR(loss='Softmax', transform='Randomization', **kw),
        'IR50-Softmax-JPEG': lambda **kw: IR(loss='Softmax', transform='JPEG', **kw),
        'IR50-PGDSoftmax': lambda **kw: IR(loss='PGDSoftmax', **kw),
        'IR50-TradesSoftmax': lambda **kw: IR(loss='TradesSoftmax', **kw),
        'IR50-CosFace': lambda **kw: IR(loss='CosFace', **kw),
        'IR50-TradesCosFace': lambda **kw: IR(loss='TradesCosFace', **kw),
        'IR50-PGDCosFace': lambda **kw: IR(loss='PGDCosFace', **kw),
        'IR50-Am': lambda **kw: IR(loss='Am', **kw),
        'IR50-PGDAm': lambda **kw: IR(loss='PGDAm', **kw),
        'IR50-ArcFace': lambda **kw: IR(loss='ArcFace', **kw),
        'IR50-PGDArcFace': lambda **kw: IR(loss='PGDArcFace', **kw),
        'IR50-TradesArcFace': lambda **kw: IR(loss='TradesArcFace', **kw),
        'IR50-SphereFace': lambda **kw: IR(loss='SphereFace', **kw),
        'IR50-PGDSphereFace': lambda **kw: IR(loss='PGDSphereFace', **kw),
        'CASIA-Softmax': lambda **kw: IR(loss='CASIA-Softmax', **kw),
        'CASIA-CosFace': lambda **kw: IR(loss='CASIA-CosFace', **kw),
        'CASIA-ArcFace': lambda **kw: IR(loss='CASIA-ArcFace', **kw),
        'CASIA-SphereFace': lambda **kw: IR(loss='CASIA-SphereFace', **kw),
        'CASIA-Am': lambda **kw: IR(loss='CASIA-Am', **kw)
    }
    return ir_models


if __name__ == "__main__":
    # Example usage
    model_name = "MobileFace"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_shape = getmodel(model_name, device=device)
    print(f"Model '{model_name}' initialized on device '{device}' with image shape {img_shape}.")
