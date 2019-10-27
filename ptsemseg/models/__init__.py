import copy
import torchvision.models as models

from ptsemseg.models.hardnet import hardnet

def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "hardnet": hardnet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
