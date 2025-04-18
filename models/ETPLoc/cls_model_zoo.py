
from models.ETPLoc.cls import (
    ETPCls,
    ETP_cls_l1,
    ETP_cls_l2,
    ETP_cls_l3,
    ETP_cls_cl0,
    ETP_cls_cl1,
    ETP_cls_cl2,
    ETP_cls_cl3,
    ETP_cls_featureAdd234_cl1,
    ETP_cls_featureAdd324_cl1,
    ETP_cls_featureAdd432_cl1,
    ETP_cls_featureAdd234_cl2,
    ETP_cls_featureAdd324_cl2,
    ETP_cls_featureAdd432_cl2,
    ETP_cls_featureAdd324_cl3,
)
from models.ETPLoc.nn.norm import set_norm_eps

__all__ = ["create_cls_model"]


def create_cls_model(name: str, **kwargs) -> ETPCls:
    model_dict = {
        "l1": ETP_cls_l1,
        "l2": ETP_cls_l2,
        "l3": ETP_cls_l3,
        #########################
        "cl0": ETP_cls_cl0,
        "cl1": ETP_cls_cl1,
        "cl2": ETP_cls_cl2,
        "cl3": ETP_cls_cl3,
        "fa_1_cl1": ETP_cls_featureAdd234_cl1,
        "fa_2_cl1": ETP_cls_featureAdd324_cl1,
        "fa_4_cl1": ETP_cls_featureAdd432_cl1,
        "fa_1_cl2": ETP_cls_featureAdd234_cl2,
        "fa_2_cl2": ETP_cls_featureAdd324_cl2,
        "fa_4_cl2": ETP_cls_featureAdd432_cl2,
        "fa_2_cl3": ETP_cls_featureAdd324_cl3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
    set_norm_eps(model, 1e-7)

    return model
