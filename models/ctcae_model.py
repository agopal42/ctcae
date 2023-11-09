from typing import Tuple

import math
import torch
import torch.nn as nn
from einops import rearrange, reduce

from models import ctcae_eval
from models import networks
import utils
import utils_contrastive


class ComplexAutoEncoderV2(nn.Module):
    def __init__(self,
    img_resolution: Tuple[int, int],
    n_in_channels: int,
    enc_n_out_channels: str,
    enc_strides: str,
    enc_kernel_sizes: str,
    d_linear: int,
    decoder_type: str,
    use_out_conv: bool,
    use_out_sigmoid: bool,
    phase_init: str,
    use_contrastive_loss: bool,
    contrastive_temperature: float,
    # Encoder's last layer contrastive loss params
    enc_n_anchors_to_sample: int,
    enc_n_positive_pairs: int,
    enc_n_negative_pairs: int,
    enc_contrastive_top_k: int,
    enc_contrastive_bottom_m: int,
    # Decoder's last layer contrastive loss params
    dec_n_anchors_to_sample: int,
    dec_n_positive_pairs: int,
    dec_n_negative_pairs: int,
    dec_contrastive_top_k: int,
    dec_contrastive_bottom_m: int,
    dec_use_patches: int,
    dec_patch_size: int,
    seed: int,
    ):
        super(ComplexAutoEncoderV2, self).__init__()

        # The below 2 options are saved for the evaluation
        self.img_resolution = img_resolution
        self.seed = seed

        enc_n_out_channels = [int(i) for i in enc_n_out_channels.split(',')]
        enc_strides = [int(i) for i in enc_strides.split(',')]
        enc_kernel_sizes = [int(i) for i in enc_kernel_sizes.split(',')]
        assert len(enc_n_out_channels) == len(enc_strides), f'Encoder output channels and strides lists need to have the same length, got {enc_n_out_channels} and {enc_strides}'
        assert len(enc_n_out_channels) == len(enc_kernel_sizes), f'Encoder output channels and strides lists need to have the same length, got {enc_n_out_channels} and {enc_kernel_sizes}'
        self.phase_init = phase_init
        self.decoder_type = decoder_type
        self.use_out_conv = use_out_conv
        self.use_out_sigmoid = use_out_sigmoid

        self.use_contrastive_loss = use_contrastive_loss

        self.enc_n_anchors_to_sample = enc_n_anchors_to_sample
        self.enc_n_positive_pairs = enc_n_positive_pairs
        self.enc_n_negative_pairs = enc_n_negative_pairs
        self.enc_contrastive_top_k = enc_contrastive_top_k
        self.enc_contrastive_bottom_m = enc_contrastive_bottom_m

        self.dec_n_anchors_to_sample = dec_n_anchors_to_sample
        self.dec_n_positive_pairs = dec_n_positive_pairs
        self.dec_n_negative_pairs = dec_n_negative_pairs
        self.dec_contrastive_top_k = dec_contrastive_top_k
        self.dec_contrastive_bottom_m = dec_contrastive_bottom_m
        self.dec_use_patches = dec_use_patches
        self.dec_patch_size = dec_patch_size

        self.encoder = ComplexEncoder(
            img_resolution=img_resolution,
            n_in_channels=n_in_channels,
            enc_n_out_channels=enc_n_out_channels,
            enc_strides=enc_strides,
            enc_kernel_sizes=enc_kernel_sizes,
            d_linear=d_linear,
        )
        self.decoder = ComplexConvDecoder(
            decoder_type=self.decoder_type,
            n_in_channels=n_in_channels,
            enc_n_out_channels=enc_n_out_channels,
            enc_strides=enc_strides,
            enc_kernel_sizes=enc_kernel_sizes,
            d_linear=d_linear,
            enc_output_res=self.encoder.enc_output_res,
        )
        if self.use_out_conv:
            self.output_model = nn.Conv2d(n_in_channels, n_in_channels, 1, 1)
            self._init_output_model()
        
        self.contrastive_temperature = contrastive_temperature
    
    def get_metrics_recorder(self):
        return ctcae_eval.CAEMetricsRecorder()
    
    def get_contrastive_temperature_scalar(self):
        if type(self.contrastive_temperature) == nn.Parameter:
            return self.contrastive_temperature.detach().cpu().numpy()
        else:
            return self.contrastive_temperature

    def _init_output_model(self):
        nn.init.constant_(self.output_model.weight, 1)
        nn.init.constant_(self.output_model.bias, 0)

    def _prepare_input(self, input_images, phase_init="zero"):
        if phase_init == "zero":
            phase = torch.zeros_like(input_images)
        elif phase_init == "random":
            phase = torch.rand_like(input_images)
        complex_input = input_images * torch.exp(phase * 1j)
        return complex_input

    def _process_layer_maps(self, l_maps, img_size):
        layer_maps = []
        for map_l in l_maps:
            # average map (across channels)
            avg_map_l = reduce(map_l, 'b c h w -> b 1 h w', 'mean')
            # upsample magnitude & phase maps if smaller than img_size
            if avg_map_l.shape[2] != img_size:
                up_avg_map_l = torch.nn.functional.interpolate(avg_map_l, img_size, mode='bilinear', align_corners=False)
            layer_maps.append(up_avg_map_l)

        layer_maps = rearrange(layer_maps, 'l b c h w -> b l c h w')
        return layer_maps 

    def _apply_module(self, z, module, channel_norm):
        m, phi = module(z)
        z = self._apply_activation_function(m, phi, channel_norm)
        return z

    def _apply_activation_function(self, m, phi, channel_norm):
        m = channel_norm(m)
        m = torch.nn.functional.relu(m)
        complex_act = m * torch.exp(phi * 1j)
        return complex_act

    def _apply_conv_layers(self, model, z):
        layer_activation_maps = []
        for idx, _ in enumerate(model.conv_layers):
            z = self._apply_module(z, model.conv_layers[idx], model.conv_norm_layers[idx])
            layer_activation_maps.append(z)
        return z, layer_activation_maps

    def encode(self, x):
        self.batch_size = x.size()[0]

        z, enc_layer_activation_maps = self._apply_conv_layers(self.encoder, x)

        if self.encoder.linear_output_layer is not None:
            z = rearrange(z, "b c h w -> b (c h w)")
            z = self._apply_module(z, self.encoder.linear_output_layer, self.encoder.linear_output_norm)
        return z, enc_layer_activation_maps

    def decode(self, z):
        if self.decoder.linear_input_layer is not None:
            z = self._apply_module(z, self.decoder.linear_input_layer, self.decoder.linear_input_norm)
            z = rearrange(z, "b (c h w) -> b c h w", b=self.batch_size, h=self.decoder.enc_output_res[0], w=self.decoder.enc_output_res[1])

        complex_output, dec_layer_activation_maps = self._apply_conv_layers(self.decoder, z)

        reconstruction = complex_output.abs()
        if self.use_out_conv:
            reconstruction = self.output_model(reconstruction)
        if self.use_out_sigmoid:
            reconstruction = torch.sigmoid(reconstruction)

        return reconstruction, complex_output, dec_layer_activation_maps

    def forward(self, input_images, step_number):
        complex_input = self._prepare_input(input_images, self.phase_init)

        z, enc_layer_activation_maps = self.encode(complex_input)
        reconstruction, complex_output, dec_layer_activation_maps = self.decode(z)

        # Processing Encoder layer(s) complex activation map
        enc_layer_phase_maps = map(lambda x: x.angle(), enc_layer_activation_maps)
        enc_layer_phase_maps = self._process_layer_maps(
            enc_layer_phase_maps, input_images.shape[2]
        )
        enc_layer_magnitude_maps = map(lambda x: x.abs(), enc_layer_activation_maps)
        enc_layer_magnitude_maps = self._process_layer_maps(
            enc_layer_magnitude_maps, input_images.shape[2]
        )
        # Processing Decoder layer(s) complex activation map
        dec_layer_phase_maps = map(lambda x: x.angle(), dec_layer_activation_maps)
        dec_layer_phase_maps = self._process_layer_maps(
            dec_layer_phase_maps, input_images.shape[2]
        )
        dec_layer_magnitude_maps = map(lambda x: x.abs(), dec_layer_activation_maps)
        dec_layer_magnitude_maps = self._process_layer_maps(
            dec_layer_magnitude_maps, input_images.shape[2]
        )

        outputs = {
            "reconstruction": reconstruction,
            "complex_output": complex_output,
            "complex_magnitude": complex_output.abs(),
            "complex_phase": complex_output.angle(),
            "enc_layer_phase_maps": enc_layer_phase_maps,
            "enc_layer_magnitude_maps": enc_layer_magnitude_maps,
            "dec_layer_phase_maps": dec_layer_phase_maps,
            "dec_layer_magnitude_maps": dec_layer_magnitude_maps,
        }

        # Contrastive loss
        if self.use_contrastive_loss:
            
            anchors = []
            positive_pairs = []
            negative_pairs = []

            assert self.enc_n_anchors_to_sample > 0 or self.dec_n_anchors_to_sample > 0, "At least one of the encoder or decoder should have a positive number of anchors to sample to use for the contrastive loss."

            if self.enc_n_anchors_to_sample > 0:
                enc_last_layer_activation_map = enc_layer_activation_maps[-1]
                enc_anchors, enc_positive_pairs, enc_negative_pairs = utils_contrastive.get_anchors_positive_and_negative_pairs(
                    addresses=enc_last_layer_activation_map.abs(),
                    features=enc_last_layer_activation_map.angle(),
                    n_anchors_to_sample=self.enc_n_anchors_to_sample,
                    n_positive_pairs=self.enc_n_positive_pairs,
                    n_negative_pairs=self.enc_n_negative_pairs,
                    top_k=self.enc_contrastive_top_k,
                    bottom_m=self.enc_contrastive_bottom_m,
                    use_patches=False,
                    patch_size=-1,
                )
                anchors.append(enc_anchors)
                positive_pairs.append(enc_positive_pairs)
                negative_pairs.append(enc_negative_pairs)

            if self.dec_n_anchors_to_sample > 0:
                dec_last_layer_activation_map = dec_layer_activation_maps[-1]
                dec_anchors, dec_positive_pairs, dec_negative_pairs = utils_contrastive.get_anchors_positive_and_negative_pairs(
                    addresses=dec_last_layer_activation_map.abs(),
                    features=dec_last_layer_activation_map.angle(),
                    n_anchors_to_sample=self.dec_n_anchors_to_sample,
                    n_positive_pairs=self.dec_n_positive_pairs,
                    n_negative_pairs=self.dec_n_negative_pairs,
                    top_k=self.dec_contrastive_top_k,
                    bottom_m=self.dec_contrastive_bottom_m,
                    use_patches=self.dec_use_patches,
                    patch_size=self.dec_patch_size,
                )
                anchors.append(dec_anchors)
                positive_pairs.append(dec_positive_pairs)
                negative_pairs.append(dec_negative_pairs)

            outputs['anchors'] = anchors
            outputs['positive_pairs'] = positive_pairs
            outputs['negative_pairs'] = negative_pairs

        return outputs
    

def init_conv_norms(conv_shapes):
    # conv_shape = (C, H, W) - if using BatchNorm, take only the first (C) dimension
    channel_norm = nn.ModuleList([None] * len(conv_shapes))
    for idx, conv_shape in enumerate(conv_shapes):
        channel_norm[idx] = nn.BatchNorm2d(conv_shape[0], affine=True)

    return channel_norm


def get_conv_biases(n_out_channels, fan_in):
    magnitude_bias = nn.Parameter(torch.empty((1, n_out_channels, 1, 1)))
    magnitude_bias = init_magnitude_bias(fan_in, magnitude_bias)

    phase_bias = nn.Parameter(torch.empty((1, n_out_channels, 1, 1)))
    phase_bias = init_phase_bias(phase_bias)
    return magnitude_bias, phase_bias


def init_magnitude_bias(fan_in, bias):
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)
    return bias


def init_phase_bias(bias):
    return nn.init.constant_(bias, val=0)


def get_stable_angle(x: torch.tensor, eps=1e-8):
    """ Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return y.angle()


def apply_layer(x, real_function, phase_bias=None, magnitude_bias=None):
    
    psi = real_function(x.real) + 1j * real_function(x.imag)

    m_psi = psi.abs()
    phi_psi = get_stable_angle(psi)
    chi = real_function(x.abs())

    if magnitude_bias is not None:
        m_psi = m_psi + magnitude_bias
        chi = chi + magnitude_bias
    if phase_bias is not None:
        phi_psi = phi_psi + phase_bias

    m = 0.5 * m_psi + 0.5 * chi

    return m, phi_psi


class ComplexLinear(nn.Module):
    def __init__(self, n_in_channels, n_out_channels):
        super(ComplexLinear, self).__init__()
        self.fc = nn.Linear(n_in_channels, n_out_channels, bias=False)
        self.magnitude_bias, self.phase_bias = self._get_biases(n_in_channels, n_out_channels)

    def _get_biases(self, n_in_channels, n_out_channels):
        fan_in = n_in_channels
        magnitude_bias = nn.Parameter(torch.empty((1, n_out_channels)))
        magnitude_bias = init_magnitude_bias(fan_in, magnitude_bias)

        phase_bias = nn.Parameter(torch.empty((1, n_out_channels)))
        phase_bias = init_phase_bias(phase_bias)
        return magnitude_bias, phase_bias

    def forward(self, x):
        return apply_layer(x, self.fc, self.phase_bias, self.magnitude_bias)


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        n_grid_directions = 4
        self.dense = nn.Linear(in_features=n_grid_directions, out_features=hidden_size)
        self.register_buffer("grid", utils.build_grid(resolution))

    def forward(self, inputs: torch.Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        res = inputs + emb_proj
        return res


class ComplexConv2d(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size=3, stride=1):
        super(ComplexConv2d, self).__init__()
        self.conv = networks.Conv2dTF(n_in_channels, n_out_channels, kernel_size, (stride, stride), bias=False)
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = n_in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = get_conv_biases(n_out_channels, fan_in)

    def forward(self, x):
        return apply_layer(x, self.conv, self.phase_bias, self.magnitude_bias)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(n_in_channels, n_out_channels, kernel_size, stride, padding, output_padding, bias=False)
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = n_out_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = get_conv_biases(n_out_channels, fan_in)

    def forward(self, x):
        return apply_layer(x, self.conv, self.phase_bias, self.magnitude_bias)


class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor: Tuple[float, float]):
        super(ComplexUpsample, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        up_x_real = self.upsample(x.real)
        up_x_imag = self.upsample(x.imag)
        return up_x_real + 1j * up_x_imag


class ComplexConvUpsample2d(ComplexConv2d):
    def __init__(self, n_in_channels, n_out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ComplexConvUpsample2d, self).__init__(n_in_channels, n_out_channels, kernel_size, 1)
        self.upsample = ComplexUpsample((stride, stride))

    def forward(self, x):
        x = self.upsample(x)
        return super().forward(x)


def get_conv_layers_shapes(conv_layers, d_channels, d_height, d_width):
    x = torch.zeros(1, d_channels, d_height, d_width)
    x = x + 1j*x
    shapes = []
    for module in conv_layers:
        m, phi = module(x)
        x = m * torch.exp(phi * 1j)
        shapes.append(x.real.shape[1:])
    return shapes


class ComplexEncoder(nn.Module):
    def __init__(
        self,
        img_resolution: Tuple[int, int], 
        n_in_channels: int,
        enc_n_out_channels: Tuple[int, ...],
        enc_strides: Tuple[int, ...],
        enc_kernel_sizes: Tuple[int, ...],
        d_linear: int,
    ):
        super().__init__()

        enc_n_out_channels = [n_in_channels] + enc_n_out_channels
        self.conv_layers = []

        for i in range(len(enc_n_out_channels) - 1):
            self.conv_layers.append(
                ComplexConv2d(
                    enc_n_out_channels[i],
                    enc_n_out_channels[i+1],
                    kernel_size=enc_kernel_sizes[i],
                    stride=enc_strides[i],
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        conv_shapes = get_conv_layers_shapes(self.conv_layers, n_in_channels, img_resolution[0], img_resolution[1])
        self.enc_output_res = conv_shapes[-1][-2:]
        self.conv_norm_layers = init_conv_norms(conv_shapes)

        d_enc_output = self.enc_output_res[0] * self.enc_output_res[1] * enc_n_out_channels[-1]
        self.linear_output_layer = None
        self.linear_output_layer = ComplexLinear(d_enc_output, d_linear)
        self.linear_output_norm = nn.LayerNorm(d_linear, elementwise_affine=True)


class ComplexConvDecoder(nn.Module):
    def __init__(
        self,
        decoder_type: str,
        n_in_channels: int,
        enc_n_out_channels: Tuple[int, ...],
        enc_strides: Tuple[int, ...],
        enc_kernel_sizes: Tuple[int, ...],
        d_linear: int,
        enc_output_res: Tuple[int, int],
    ):
        super().__init__()

        self.enc_output_res = enc_output_res
        dec_n_out_channels = enc_n_out_channels[::-1]
        dec_strides = enc_strides[::-1]
        dec_kernel_sizes = enc_kernel_sizes[::-1]
        dec_n_out_channels = dec_n_out_channels + [n_in_channels]

        if decoder_type == 'conv_transpose':
            conv_upsample_cls = ComplexConvTranspose2d
            for i in range(len(dec_kernel_sizes)):
                if dec_kernel_sizes[i] != 3:
                    print(f'Warning: resetting the kernel size from {dec_kernel_sizes[i]} to 3 since automatic padding calculation for random kernel sizes is not implemented for ConvTranspose.')
                    dec_kernel_sizes[i] = 3
        elif decoder_type == 'conv_upsample':
            conv_upsample_cls = ComplexConvUpsample2d

        self.linear_input_layer = None
        d_linear_out = dec_n_out_channels[0] * self.enc_output_res[0] * self.enc_output_res[1]
        self.linear_input_layer = ComplexLinear(d_linear, d_linear_out)
        self.linear_input_norm = nn.LayerNorm(d_linear_out, elementwise_affine=True)

        self.conv_layers = []
        for i in range(len(dec_n_out_channels) - 1):
            if dec_strides[i] == 1:
                self.conv_layers.append(
                    ComplexConv2d(
                        dec_n_out_channels[i],
                        dec_n_out_channels[i+1],
                        kernel_size=dec_kernel_sizes[i],
                        stride=dec_strides[i],
                    )
                )
            else:
                self.conv_layers.append(
                    conv_upsample_cls(
                        dec_n_out_channels[i],
                        dec_n_out_channels[i+1],
                        kernel_size=dec_kernel_sizes[i],
                        output_padding=1,
                        padding=1,
                        stride=dec_strides[i],
                    )
                )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        conv_shapes = get_conv_layers_shapes(self.conv_layers, dec_n_out_channels[0], self.enc_output_res[0], self.enc_output_res[1])
        self.conv_norm_layers = init_conv_norms(conv_shapes)
