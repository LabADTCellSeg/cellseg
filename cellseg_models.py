from typing import Optional, Union, List
import copy

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.initialization as init


class CustomUNetWithSeparateDecoderForBoundary(nn.Module):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 classes: int = 1,
                 boundary_classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None
                 ):
        super().__init__()

        # Общий энкодер из SMP
        self.encoder = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                in_channels=in_channels, classes=classes).encoder

        # Декодер для клеток
        self.decoder_cells = smp.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        # Декодер для границ
        self.decoder_boundaries = smp.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
        )

        # Сегментационная голова для клеток (multiclass)
        self.segmentation_head_cells = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=classes,  # Классы клеток
            activation=activation,
            kernel_size=3
        )

        # Сегментационная голова для границ (один класс)
        self.segmentation_head_boundaries = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=boundary_classes,  # Один класс для границ
            activation=activation,
            kernel_size=3
        )

    def forward(self, x):
        # Пропускаем входные данные через энкодер
        features = self.encoder(x)

        # Пропускаем через декодер клеток
        decoder_output_cells = self.decoder_cells(*features)
        mask_cells = self.segmentation_head_cells(decoder_output_cells)

        # Пропускаем через декодер границ
        decoder_output_boundaries = self.decoder_boundaries(*features)
        mask_boundaries = self.segmentation_head_boundaries(decoder_output_boundaries)

        # Объединяем выходы
        return mask_cells, mask_boundaries

    def initialize(self):
        init.initialize_decoder(self.decoder_cells)
        init.initialize_decoder(self.decoder_boundaries)
        init.initialize_head(self.segmentation_head_cells)
        init.initialize_head(self.segmentation_head_boundaries)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)


def create_model_with_separate_decoder_for_boundary(model_name,
                                                    encoder_name,
                                                    encoder_weights,
                                                    in_channels,
                                                    classes,
                                                    boundary_classes,
                                                    activation,
                                                    decoder_channels=(256, 128, 64, 32, 16)
                                                    ):
    import importlib
    from segmentation_models_pytorch.base import (
        SegmentationHead
    )
    from types import MethodType

    model_fn = getattr(importlib.import_module('segmentation_models_pytorch'), model_name)
    model = model_fn(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        decoder_channels=decoder_channels
    )

    if boundary_classes >= 1:
        model.boundary_classes = boundary_classes
        model.decoder_boundaries = copy.deepcopy(model.decoder)

        model.segmentation_head_boundaries = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=boundary_classes,
            activation=activation,
            kernel_size=3,
        )

        init.initialize_decoder(model.decoder_boundaries)
        init.initialize_head(model.segmentation_head_boundaries)

    def forward(self, x):
        # Пропускаем входные данные через энкодер
        features = self.encoder(x)

        # Пропускаем через декодер клеток
        decoder_output_cells = self.decoder(*features)
        masks = self.segmentation_head(decoder_output_cells)

        if self.boundary_classes >= 1:
            # Пропускаем через декодер границ
            decoder_output_boundaries = self.decoder_boundaries(*features)
            mask_boundaries = self.segmentation_head_boundaries(decoder_output_boundaries)
            masks = torch.cat([masks, mask_boundaries], dim=1)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        # Объединяем выходы
        return masks

    model.forward = MethodType(forward, model)

    return model