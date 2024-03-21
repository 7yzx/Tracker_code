from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        # z是模版，x是搜索区域
        # nz=8 bs
        nz = z.size(0) # nz是模版的batchsize ,
        # 8,256,20,20
        nx, c, h, w = x.size() #
        # 这里的第二个维度为了将搜索图像中的通道与模板图像 z 进行分组卷积时的输入形状匹配。
        # 1,2048,20,20
        x = x.view(-1, nz * c, h, w)
        # 每个组对应 z 中的一个通道。这实际上是在每个位置上计算搜索图像和模板图像之间的相似性。
        # 1,8,15,15
        out = F.conv2d(x, z, groups=nz)
        # 卷积操作的输出重新形状回原始形状，其中第二个维度被调整为 -1，以保持数据的一致性。
        # 这样做后，out 的每个元素表示在搜索图像中的相应位置上，与模板图像最相似的程度。
        # 8,1,15,15
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
