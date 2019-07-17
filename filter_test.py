import mxnet as mx
import numpy as np
import os, time, logging, math, argparse

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


def try_gpu():
    """
    如果gpu可用，返回gpu，否则返回cpu。
    If gpu is availle, return gpu; otherwise return cpu.

    :return : mx.gpu(0)/mx.cpu(0)
    """
    try:
        ctx = mx.gpu(0)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu(0)
    return ctx


def filter_image(data, filter_level):
    """
    对图片的像素值进行过滤处理。
    Filter pixel values of image.

    :param data: image data, mx.ndarray
    :param filter_level: filter level, int
    :return data: filtered image data, mx.ndarray
    """
    data = data.asnumpy()
    data = data // (2**filter_level)
    data = data * (2**filter_level)
    data = nd.array(data)
    return data


RESIZE_SIZE = 224 # resize short 
INPUT_SIZE = 224 # center crop 
def get_tranform_func(filter_level):
    """
    获取具有不同过滤级别的前处理函数。
    Get transform function with different levels of filtering.

    :param filter_level: filter level, int
    :return transform_filter: transform function for image pre-processing, function
    """
    def transform_filter(data, label):
        """
        用于图片/标签前处理的函数。
        Transform function for image/label pre-processing.

        :param data: image data, mx.ndarray
        :param label: image label, mx.ndarray
        """
        im = filter_image(data, filter_level) # compress image to certain bit
        im = im.astype("float32") / 255
        im = image.resize_short(im, RESIZE_SIZE)
        im, _ = image.center_crop(im, (INPUT_SIZE, INPUT_SIZE))
        im = nd.transpose(im, (2,0,1))
        im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return (im, nd.array([label]).asscalar())
    
    return transform_filter


def progressbar(i, n, bar_len=40):
    """
    显示测试的进度。
    Show progress of test.

    :param i: progress rate, int
    :param n: total progress length, int
    :param bar_len: progress bar length, int
    """
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s' % (prog_bar, percents, '%'), end = '\r')


def validate(net, val_data, ctx):
    """
    计算预训练的模型在数据集上top-1/top-5准确率。
    Calculate top-1/top-5 accuracy of pretrained net on dataset.

    :param net: network
    :param val_data: dataset being tested on
    :param ctx: computing context
    :return val_acc_top1: top-1 accuracy, float
    :return val_acc_top5: top-5 accuracy, float
    """
    metric = mx.metric.Accuracy()
    metric_5 = mx.metric.TopKAccuracy(5)
    batch_length = len(val_data)

    for i, batch in enumerate(val_data):
        data = batch[0].as_in_context(ctx)
        label = batch[1].as_in_context(ctx)
        output = net(data)
                
        metric.update(labels=label, preds=output)
        metric_5.update(labels=label, preds=output)
        progressbar(i, batch_length)

    _, val_acc_top1 = metric.get()
    _, val_acc_top5 = metric_5.get()
    return val_acc_top1, val_acc_top5


def filter_test(model_name, dataset_dir, batch_size=16, num_workers=8):
    """
    在不同的过滤级别下，对MXNet预训练模型在imagenet数据集上的准确率进行测试。
    Test accuracy of MXNet pretrained model on imagenet dataset with 
    different filter levels.

    :param model_name: model name in model_zoo, str
    :param dataset_dir: image dataset directory, str
    :param batch_size: batch size, int
    :param num_workers: number of image processing workers, int
    """
    # get ctx, net
    ctx = try_gpu()
    net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # valiadate net on imagenet dataset with different filter levels
    for filter_level in range(1, 8):
        transform_func = get_tranform_func(filter_level)
        val_data = gluon.data.DataLoader(
                    gluon.data.vision.ImageFolderDataset(dataset_dir,
                    transform=transform_func), 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=num_workers)
        acc_top1, acc_top5 = validate(net, val_data, ctx)
        print("Filter level %d, Top-1 accuracy %.3f, Top-5 accuracy %.3f" % (filter_level, acc_top1, acc_top5))


if __name__ == "__main__":
    model_name = "resnet34_v2"
    dataset_dir = "ImageNet/val"
    batch_size = 64
    num_workers = 8
    filter_test(model_name, dataset_dir, batch_size=16, num_workers=8)
