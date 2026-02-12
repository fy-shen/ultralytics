import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import os
import cv2
import math
import copy
import numpy as np


class MSTF(nn.Module):
    def __init__(self, in_channel, hidden_dim=64, epoch_train=22, stride=[2, 2, 2], radius=[3, 3, 3], n_levels=3,
                 iter_max=2, method="method1", motion_flow=True, aux_loss=False):
        super(MSTF, self).__init__()
        input_dim = in_channel[0]

        if not (input_dim // 2 >= hidden_dim):
            print("************warning****************")
            print(f"input_dim//2 need bigger than hidden_dim, {in_channel},{hidden_dim}")
            print("***********************************")

        self.in_channel = in_channel

        self.hidden_dim = hidden_dim  # input_dim//2
        self.iter_max = iter_max
        self.n_levels = n_levels
        self.radius = radius
        self.stride = stride
        self.epoch_train = epoch_train
        self.method = method
        self.aux_loss = aux_loss
        self.motion_flow = motion_flow
        self.cor_planes = [n_levels * (2 * radiu + 1) ** 2 for radiu in radius]

        self.convs1 = nn.ModuleList([FlowConv(in_channel[i], self.hidden_dim, 1)
                                     for i in range(n_levels)])

        self.convs2 = nn.ModuleList([FlowConv(self.hidden_dim + self.hidden_dim // 2, in_channel[i], 1)
                                     for i in range(n_levels)])

        # buffer
        self.buffer = FlowBuffer("MemoryAtten", number_feature=n_levels)

        cor_plane = self.cor_planes[1]
        self.cor_plane = cor_plane
        self.cor_plane = 2 * (self.cor_plane // 2)  # Guaranteed to be even.

        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane, self.cor_planes[0], self.cor_plane)
        self.flow_fused2 = FlowDown(self.cor_plane, self.cor_plane, self.cor_planes[2], self.cor_plane)

        self.update_block = SmallNetUpdateBlock(
            input_dim=self.hidden_dim // 2, hidden_dim=self.hidden_dim // 2, cor_plane=self.cor_plane
        )

        self.plot = False
        self.save_dir = "./MSTF_saveDir"
        self.pad_image_func = None

    def forward(self, x):
        with autocast(dtype=torch.float32):
            # x 是 0,6,8,11 的输出
            # 0是原始输入，主要用到数据的信息
            # 后三项构成特征金字塔，是MSTF核心使用的数据
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]

            # 训练初期 MSTF 不使用时序，只是简单用卷积处理各层特征图
            if fmaps_new[0].device.type == "cpu" or (
                    img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:]
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    out = self.convs2[i](torch.cat([out, fmap, torch.relu(fmap)], 1))
                    outs.append(out)
                return outs

            # Gradient triage, dimensional consistency
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))

            if not self.training and self.plot:
                video_name = img_metas[0]["video_name"]
                save_dir = os.path.join(self.save_dir, video_name)
                os.makedirs(save_dir, exist_ok=True)
                image_path = img_metas[0]["image_path"]

                image = cv2.imread(image_path)
                image, _ = self.pad_image_func(image)
                height, width, _ = image.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                if img_metas[0]["is_first"]:
                    if hasattr(self, "video_writer_fea"):
                        self.video_writer_fea.release()
                    self.number = 0
                    self.video_writer_fea = cv2.VideoWriter(
                        os.path.join(save_dir, "feature_fused.mp4"), fourcc, 25, (width, height)
                    )

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(
                src_flatten_new, img_metas, spatial_shapes, level_start_index
            )

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(
                    AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride)
                )

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]

            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(corr_32, corr_16)
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1 / 4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1 / 2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            # get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)

            for i in range(self.n_levels):
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i], fmaps_new[i], net[i]], 1)) + x[1:][i]

            if not self.training and self.plot:
                overlay_heatmap_on_video(self.video_writer_fea, image, fmaps_new)
                self.number += 1

            return fmaps_new


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, stride=None):
        if stride is None:
            stride = [1, 1, 1]

        self.num_levels = num_levels
        self.radius = radius
        self.stride = stride

        self.pyramid = []
        for i in range(self.num_levels):
            self.pyramid.append((fmap1, fmap2[i]))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            # fmap1_i: 初始化输入的 fmap1
            # fmap2_i: 不同分辨率的特征图 p3->p4->p5
            # coords:  常见的生成坐标 b,h,w,2 这里没有归一化
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()
            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            # 核心计算，输出 b,1,81,h,w
            corr, = alt_cuda_corr.forward(fmap1_i.float(), fmap2_i.float(), coords_i.float(), self.radius, self.stride[i])
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)  # b,3,81,h,w
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class FlowBuffer(object):
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance

    def __init__(self, name, number_feature=3):
        super().__init__()
        if not hasattr(self, 'initialized'):
            self.name = name
            self.bs = 0
            self.memory_fmaps = None
            self.memory_bbox = None
            self.memory_score = None
            self.img_metas_memory = [None for _ in range(self.bs)]
            self.initialized = True
            self.number_feature = number_feature
            self.coords0 = None
            self.coords1 = None
            self.spatial_shapes = None
            self.net = None

    def __reduce__(self):
        return self.__class__, (self.name, self.number_feature)

    def update_coords(self, coords1):
        self.coords1 = self.flatten([coords.detach() for coords in coords1])

    def update_net(self, nets):
        self.net = self.flatten([net.detach() for net in nets])

    def update_bbox(self, bbox, score):
        self.memory_bbox = bbox.detach()
        self.memory_score = score.detach()

    def update_memory(self, memory_flatten, img_metas, spatial_shapes, level_start_index):
        b, _, dim = memory_flatten.shape
        assert len(img_metas) == b

        if b == self.bs:  # Store current memory and image information, return previous storage
            pass
        else:
            self.reset_all(b)

        # Initialized when coords_orige is None or inconsistent.
        if self.spatial_shapes is None or (not torch.equal(self.spatial_shapes, spatial_shapes)):
            self.coords0 = self.initialize_point(self.bs, spatial_shapes)
            self.coords0 = self.flatten(self.coords0)
            self.spatial_shapes = spatial_shapes.clone()
            self.coords1 = self.coords0.clone()

            self.net = torch.tanh(memory_flatten).detach()
            self.memory_fmaps = memory_flatten.detach()

        result_first_frame = [img_metas[i]["is_first"] for i in range(b)]

        if self.memory_fmaps is None:
            self.memory_fmaps = memory_flatten.detach()
            for i in range(self.bs):
                result_first_frame[i] = True

        if self.net is None:
            self.net = torch.tanh(memory_flatten).detach()

        for i in range(self.bs):
            if result_first_frame[i]:
                self.net[i] = torch.tanh(memory_flatten[i]).detach()
                self.coords1[i] = self.coords0[i]
                self.memory_fmaps[i] = memory_flatten[i].detach()

        results_coords0 = self.recover_src(self.coords0, spatial_shapes, level_start_index)
        results_coords1 = self.recover_src(self.coords1, spatial_shapes, level_start_index)
        results_memory = self.recover_src(self.memory_fmaps, spatial_shapes, level_start_index)
        result_net = self.recover_src(self.net, spatial_shapes, level_start_index)

        # save
        self.memory_fmaps = memory_flatten.detach()
        for i in range(self.bs):
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])

        return result_first_frame, results_memory, result_net, results_coords0, results_coords1, [
            self.memory_bbox.clone(), self.memory_score.clone()] if self.memory_bbox is not None else None

    def reset_all(self, batch=1):
        self.bs = batch
        self.memory_bbox = None
        self.memory_score = None
        self.memory_fmaps = None
        self.img_metas_memory = [None for _ in range(self.bs)]
        self.coords1 = None
        self.coords0 = None
        self.spatial_shapes = None
        self.net = None

    def initialize_point(self, bs, spatial_shapes):
        coords0 = []
        for lvl, shape in enumerate(spatial_shapes):
            h, w = shape
            coords0.append(coords_grid(bs, h, w, device=spatial_shapes.device))
        return coords0

    def recover_src(self, src_flatten, spatial_shapes, level_start_index):
        srcs = []
        num_levels = level_start_index.size(0)

        bs, _, dim = src_flatten.shape
        for lvl in range(num_levels):
            start_index = level_start_index[lvl].item()
            end_index = level_start_index[lvl + 1].item() if lvl + 1 < num_levels else src_flatten.size(1)
            src = src_flatten[:, start_index:end_index].transpose(1, 2).reshape(bs, dim, *spatial_shapes[lvl])
            srcs.append(src)
        return srcs

    def flatten(self, srcs, re_shape=False):
        # prepare input for encoder
        src_flatten = []
        spatial_shapes = []
        for lvl, src in enumerate(srcs):
            if re_shape:
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        if re_shape:
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            return src_flatten, spatial_shapes, level_start_index
        return src_flatten


class ConvTranspose(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, kernel_size, stride, padding, output_padding=output_padding, bias=False)
        self.norm = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class FlowConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)


class FlowUp(nn.Module):
    def __init__(self, cx, cy, cout):
        super().__init__()
        self.c = cout // 2
        half_x = math.ceil(cx / 2)
        half_x_2 = cx // 2
        self.cv1 = ConvTranspose(half_x, self.c, 3, 2, 1, 1)
        self.cv2 = FlowConv(half_x_2 + cy, self.c, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, y):
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = self.upsample(x2)
        x2 = self.cv2(torch.cat((x2, y), 1))
        return torch.cat((x1, x2), 1)


class FlowDown(nn.Module):
    def __init__(self, c_4, c_2, c, cout):
        super().__init__()
        self.c = cout // 2
        half_c_4 = math.ceil(c_4 / 2)
        half_c_2 = math.ceil(c_2 / 2)
        half_c = math.ceil(c / 2)

        half_c_4_1 = c_4 // 2
        half_c_2_1 = c_2 // 2
        half_c_1 = c // 2

        self.cv1 = FlowConv(half_c_4, self.c, 3, 2, 1)
        self.cv2 = FlowConv(half_c_4 + half_c_2, self.c, 3, 2, 1)

        self.cv3 = FlowConv(self.c + half_c, self.c, 1, 1, 0)
        self.cv4 = FlowConv(half_c_1 + half_c_4_1 + half_c_2_1, self.c, 1, 1, 0)

    def forward(self, f_4, f_2, f):
        f_4 = torch.nn.functional.avg_pool2d(f_4, 2, 1, 0, False, True)
        f4_1, f4_2 = f_4.chunk(2, 1)
        f4_1 = self.cv1(f4_1)
        f4_2 = torch.nn.functional.max_pool2d(f4_2, 3, 2, 1)

        f_2 = torch.nn.functional.avg_pool2d(f_2, 2, 1, 1, False, True)[:, :, :f4_1.shape[2], :f4_1.shape[3]]
        f2_1, f2_2 = f_2.chunk(2, 1)
        f2_1 = self.cv2(torch.cat((f2_1, f4_1), 1))

        f2_2 = torch.nn.functional.max_pool2d(f2_2, 3, 2, 1)
        f4_2 = torch.nn.functional.max_pool2d(f4_2, 3, 2, 1)

        f1, f2 = f.chunk(2, 1)

        f1 = self.cv3(torch.cat((f2_1, f1), 1))
        f2 = self.cv4(torch.cat((f2_2, f4_2, f2), 1))

        return torch.cat((f1, f2), 1)


class SmallNetUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, cor_plane=243):
        super(SmallNetUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(cor_plane, hidden_dim - 2)  # outdim = hidden_dim
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)  # B,input_dim,H,W
        inp = torch.cat([inp, motion_features], dim=1)  # B,2*input_dim,H,W
        net = self.gru(net, inp)  # B,hidden_dim,H,W
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class SmallMotionEncoder(nn.Module):
    def __init__(self, input_dim=128, output_idm=80):
        super(SmallMotionEncoder, self).__init__()
        self.convc1 = FlowConv(input_dim, 96, 1, p=0)
        self.convf1 = FlowConv(2, 64, 7, p=3)
        self.convf2 = FlowConv(64, 32, 3, p=1)
        self.conv = FlowConv(128, output_idm, 3, p=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = FlowConv(input_dim, hidden_dim, 3, p=1)
        self.conv2 = FlowConv(hidden_dim, 2, 3, p=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def overlay_heatmap_on_video(video_writer, original_image, feature_maps):
    # 获取原始图像的大小
    original_height, original_width = original_image.shape[:2]

    # 初始化处理后的图像为原始图像的副本
    processed_image = original_image.copy()

    # 如果特征图不是列表，转换为列表以统一处理
    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]

    # 上采样所有特征图并融合
    upsampled_features = []
    for feature_map in feature_maps:
        # 上采样特征图到原图大小
        feature_map = F.interpolate(feature_map, size=(original_height, original_width), mode='bilinear',
                                    align_corners=False)
        upsampled_features.append(feature_map.squeeze(0).detach())

    # 将所有特征图沿通道方向堆叠成一个张量
    combined_feature_map = torch.cat(upsampled_features, dim=0)  # (N*C, H, W)

    # 对通道进行融合，使用最大值法（也可以选择其他方法如平均值）
    combined_feature_map = torch.max(combined_feature_map, dim=0).values  # (H, W)

    # 转换为NumPy数组
    feature_map_np = combined_feature_map.cpu().numpy() if hasattr(combined_feature_map,
                                                                   'cpu') else combined_feature_map

    # 归一化特征图到0-255范围内以便转换为灰度图
    heatmap = cv2.normalize(feature_map_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = adjust_heatmap(heatmap)
    heatmap = heatmap.astype(np.uint8)

    # 将灰度图转换为伪彩色热图
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 将热图叠加到原始图像上
    processed_image = cv2.addWeighted(processed_image, 0.7, heatmap_color, 0.3, 0)

    # 将处理后的图像帧写入视频句柄
    video_writer.write(processed_image)
    # cv2.imwrite(video_writer, processed_image)
    # return processed_image


def adjust_heatmap(heatmap):
    # 计算heatmap的平均值
    mean_value = np.median(heatmap)
    # 初始化一个与heatmap相同大小的数组
    adjusted_heatmap = np.zeros_like(heatmap)
    # 对小于平均值的值进行衰减（使用平方根或其他方法）
    adjusted_heatmap[heatmap < mean_value] = np.sqrt(heatmap[heatmap < mean_value])
    # 对大于平均值的值进行增强（使用平方或其他方法）
    adjusted_heatmap[heatmap >= mean_value] = np.square(heatmap[heatmap >= mean_value])
    # 归一化到0-255范围
    adjusted_heatmap = cv2.normalize(adjusted_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return adjusted_heatmap.astype(np.uint8)


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
