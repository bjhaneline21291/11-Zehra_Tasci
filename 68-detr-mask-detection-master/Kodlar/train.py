"""
Train a DETR model
"""
#%% Setup
import random
from glob import glob
from PIL  import Image

import torch
import pytorch_lightning as pl

from src.models.detr import DETR, SetCriterion
from src.models.backbone import build_backbone
from src.models.transformer import build_transformer
from src.models.matcher import build_matcher

from dataset import ImageDataset
from torch.utils.data import DataLoader
import transforms as T

import torchvision.transforms.functional as F

#%% Parameters
class args(object):
    def __init__(self, d):
        self.__dict__ = d

args = args(dict(backbone="resnet50", dilation=True, hidden_dim=256, position_embedding='learned', masks=False,
                 aux_loss=False, dropout=0.1, nheads=8, dim_feedforward=2048, enc_layers=6, dec_layers=6, pre_norm=True,
                 set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, bbox_loss_coef=5, mask_loss_coef=1, dice_loss_coef=1,
                 eos_coef=0.1, giou_loss_coef=2,lr_backbone=1e-5, lr=1e-4, weight_decay=1e-4, num_classes=2, num_queries=100))

#%%
weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
weight_dict['loss_giou'] = args.giou_loss_coef
if args.masks:
    weight_dict["loss_mask"] = args.mask_loss_coef
    weight_dict["loss_dice"] = args.dice_loss_coef

if args.aux_loss:
    aux_weight_dict = {}
    for i in range(args.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

losses = ['labels', 'boxes', 'cardinality']
if args.masks:
    losses += ["masks"]

#%% Model setup
backbone = build_backbone(args)
transformer = build_transformer(args)
model = DETR(backbone=backbone, transformer=transformer, num_classes=args.num_classes, num_queries=args.num_queries, aux_loss=args.aux_loss)

matcher = build_matcher(args)
criterion = SetCriterion(num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)

#%% Lighning module
class DETRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        outputs = self.model(x)

        return outputs

    def configure_optimizers(self):
        param_dicts = [{"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                       {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                        "lr": args.lr_backbone}]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        return optimizer

    def training_step(self, batch, batch_nb):
        x, y = batch
        y = [{k: v for k, v in t.items()} for t in y]

        y_ = self.model(x)

        loss_dict = self.criterion(y_, y)
        loss = sum(loss_dict[k] * self.criterion.weight_dict[k]
                   for k in loss_dict.keys() if k in self.criterion.weight_dict)

        logs = {"loss": loss}

        return {"loss": loss, "log": logs}

    def train_dataloader(self):
        # Create a label map
        label_map = {"head": 0, "helmet": 1, "background": 2}

        # Image files
        images = glob("dataset/Hardhat/All/JPEGImage/*.jpg")

        # Create a transform
        transform = T.Compose([T.NormalizeBoxes(), T.Resize([300, 300]), T.ToTensor()])

        # Create a dataset
        dataset = ImageDataset(images=images, annotation_folder="dataset/Hardhat/All/Annotation",
                               label_map=label_map, transform=transform)

        return DataLoader(dataset=dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)

#%% Training
'''
if __name__ == "__main__" and True:
    model = DETRModel()
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=4)
    trainer.fit(model)

'''
#%% Evaluation
# Load the checkpoint
model = DETRModel.load_from_checkpoint(checkpoint_path="lightning_logs/version_7/checkpoints/epoch=124.ckpt")

for image in glob(f"dataset/Hardhat/Eval/*.jpg"):
    # Load and transform the image
    x = Image.open(fp=image, mode='r').convert('RGB')
    x = F.resize(img=x, size=[300, 300])
    x = F.to_tensor(x).unsqueeze(dim=0)

    y_ = model(x)

    # kKep only predictions with 0.7+ confidence
    probabilities = y_['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = y_['pred_boxes'][0, probabilities.max(-1).values > 0.7]
    classes = y_['pred_logits'][0, probabilities.max(-1).values > 0.7].softmax(-1).argmax(dim=1)

    label_map = {"head": 0, "helmet": 1, "background":2}
    colour_map = {key:"#%06x" % random.randint(0, 0xFFFFFF) for key in label_map}

    annotated_image = ImageDataset.draw_boxes( image=F.to_pil_image(x.squeeze())
                                             , labels={"boxes":boxes, "classes":classes}
                                             , label_map=label_map
                                             , colour_map=colour_map)
    annotated_image.show()
    a=0

