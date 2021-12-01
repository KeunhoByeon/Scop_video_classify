import dill
import torch


def download_model():
    model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
    torch.save(model, 'model.pth', pickle_module=dill)


def load_labels(file_path):
    kinetics_id_to_classname = {}
    with open(file_path, "r") as rf:
        for line in rf.readlines():
            id, classname = line.replace('\n', '').split(',')
            kinetics_id_to_classname[int(id)] = classname

    return kinetics_id_to_classname


def get_topk_classes(preds, topk=5):
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=topk).indices

    return pred_classes
