import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
#import torchvision

from matplotlib.lines import Line2D
from PIL import Image, ImageFont, ImageDraw, ImageOps
import cv2

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # © Daniel Gehrig, https://github.com/uzh-rpg/rpg_event_dl_utils/blob/master/pytorch/viz/tensorboard_logging.py#L250

    # draw the renderer
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    mpl.rcParams['text.antialiased'] = False
    data = torch.from_numpy(data).permute(2, 0, 1)
    plt.close()
    return data


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    # © Daniel Gehrig, https://github.com/uzh-rpg/rpg_event_dl_utils/blob/master/pytorch/viz/tensorboard_logging.py#L250
    fig, ax = plt.subplots(figsize=(10, 5))

    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            min_grads.append(p.grad.abs().min())

    ax.bar(3*np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="r")
    ax.bar(3*np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="m")
    ax.bar(3*np.arange(len(max_grads)), min_grads, alpha=0.4, lw=1, color="b")

    ax.set_xticks(range(0, 3*len(ave_grads), 3))
    labels = ax.set_xticklabels(layers)
    for l in labels:
        l.update({"rotation": "vertical"})

    ax.set_xlim(left=0, right=3*len(ave_grads))
    ax.set_ylim(bottom=1e-7, top=1e2)
    ax.set_yscale("log")  # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="m", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient', 'min-gradient'])
    return fig


def plot_classes_preds(images, targets, outputs):
    '''
    Generates matplotlib Figure with images, labels and targets.
    '''
    # adapted from: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#
    nbr_images = targets.shape[0]
    fig = plt.figure(figsize=(nbr_images * 1.5, 1.5))
    for idx in range(0, nbr_images):
        ax = fig.add_subplot(1, nbr_images, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)).astype(np.uint8))
        # matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("Pred: sp={0:.2f}, st={1:.2f}, th={2:.2f}, br={3:.2f}\n"
            "True: sp={4:.2f}, st={5:.2f}, th={6:.2f}, br={7:.2f}".format(
            outputs[idx][0], outputs[idx][1], outputs[idx][2], outputs[idx][3],
            targets[idx][0], targets[idx][1], targets[idx][2], targets[idx][3], ), fontsize=3.5)
    return fig

def add_prediction_images(images, targets, outputs):
    nbr_images = targets.shape[0]
    '''#font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
    draw = ImageDraw.Draw(img)
    #draw.text((0,0), "This is a test", (255,255,0), font=font)
    draw.text((0,0), "This is a test", (255,255,0))
    draw = ImageDraw.Draw(img)'''
    #img = cv2.putText(img, str("Hello World"), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 1), 2)
    #img = cv2.putText(img, text="hello", org=(200, 200), fontFace=3, fontScale=3,
    #                           color=(0, 0, 1), thickness=5)
    new_images = np.empty([nbr_images*2, images[0].shape[0], images[0].shape[1], images[0].shape[2]])
    height = images[0].shape[1]
    width = images[0].shape[2]

    for idx in range(0, nbr_images):
        new_images[idx*2, :, :, :] = images[idx]
        #pred_img = np.zeros_like(images[idx])
        pred_img = Image.new('RGB', (width, height), color=(1,1,1))
        draw = ImageDraw.Draw(pred_img)

        #font = ImageFont.truetype("arial.ttf", 100)
        #font = ImageFont.load_default().font
        #font = ImageFont.truetype("arial")
        text = "Pred: sp={0:.2f}, st={1:.2f}, th={2:.2f}, br={3:.2f}\n True: sp={4:.2f}, st={5:.2f}, th={6:.2f}, br={7:.2f}".format(
            outputs[idx][0], outputs[idx][1], outputs[idx][2], outputs[idx][3],
            targets[idx][0], targets[idx][1], targets[idx][2], targets[idx][3])
        # text = "Pred: sp={0:.2f}, st={1:.2f}, th={2:.2f}\n True: sp={3:.2f}, st={4:.2f}, th={5:.2f}".format(
        #    outputs[idx][0], outputs[idx][1], outputs[idx][2],
        #    targets[idx][0], targets[idx][1], targets[idx][2])

        #draw.text((0,0), str(text), (255,255,0), font=font)
        draw.text((0, 0), text, (0, 0, 255))
        draw = ImageDraw.Draw(pred_img)
        #pred_img = ImageOps.mirror(pred_img)
        #pred_img = pred_img.rotate(-90)
        #pred_img = cv2.putText(pred_img, str("Hello World"), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 1), 2)
        new_images[idx*2+1, :, :, :] = np.transpose(np.asarray(pred_img), (2, 0, 1))

    return torch.from_numpy(new_images)

