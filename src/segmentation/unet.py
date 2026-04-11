"""Definição e treinamento da U-Net MobileNetV2 para segmentação de peixes."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters, morphology
from tensorflow_examples.models.pix2pix import pix2pix

from src.commons import binarize_image, find_bighest_cluster

DIM: int = 224


@adapt_rgb(each_channel)
def merge_image_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica a máscara a cada canal da imagem multiplicativamente.

    Args:
        image: Imagem RGB de entrada.
        mask: Máscara binária correspondente.

    Returns:
        Imagem com fundo mascarado.
    """
    return image * mask


def create_mask(pred_mask: tf.Tensor) -> np.ndarray:
    """Converte a saída do modelo em uma máscara 2-D.

    Args:
        pred_mask: Tensor de predição com shape (batch, H, W, C).

    Returns:
        Array 2-D com a máscara resultante.
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return np.asarray(pred_mask).reshape(DIM, DIM)


def smoothing_mask(mask: np.ndarray) -> np.ndarray:
    """Suaviza e preenche buracos em uma máscara predita.

    Args:
        mask: Máscara binária bruta.

    Returns:
        Máscara processada (maior cluster, fechamento morfológico, preenchimento).
    """
    mask = find_bighest_cluster(mask)
    mask = morphology.closing(mask, morphology.disk(9))
    mask = ndi.binary_fill_holes(mask)
    mask = filters.gaussian(mask, sigma=0.5)
    return binarize_image(mask)


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Coeficiente de Dice entre máscara real e predita.

    Args:
        y_true: Máscara real.
        y_pred: Máscara predita.
        smooth: Fator de suavização para evitar divisão por zero.

    Returns:
        Escalar Tensor com o coeficiente de Dice.
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Coeficiente IoU (Intersection over Union) entre máscara real e predita.

    Args:
        y_true: Máscara real.
        y_pred: Máscara predita.
        smooth: Fator de suavização para evitar divisão por zero.

    Returns:
        Escalar Tensor com o coeficiente IoU.
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def unet_model(dim: int = DIM) -> tf.keras.Model:
    """Constrói o modelo U-Net com backbone MobileNetV2 pré-treinado.

    Args:
        dim: Dimensão (altura e largura) das imagens de entrada.

    Returns:
        Modelo Keras U-Net compilável.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[dim, dim, 3], include_top=False
    )

    layer_names = [
        "block_1_expand_relu",   # 64×64
        "block_3_expand_relu",   # 32×32
        "block_6_expand_relu",   # 16×16
        "block_13_expand_relu",  # 8×8
        "block_16_project",      # 4×4
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4×4  → 8×8
        pix2pix.upsample(256, 3),  # 8×8  → 16×16
        pix2pix.upsample(128, 3),  # 16×16 → 32×32
        pix2pix.upsample(64, 3),   # 32×32 → 64×64
    ]

    inputs = tf.keras.layers.Input(shape=[dim, dim, 3])
    x = inputs

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        1, 3, strides=2, padding="same", activation="sigmoid"
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=last)


def vanilla_unet(dim: int = DIM) -> tf.keras.Model:
    """Constrói uma U-Net vanilla sem backbone pré-treinado.

    Args:
        dim: Dimensão (altura e largura) das imagens de entrada.

    Returns:
        Modelo Keras U-Net vanilla.
    """

    def conv_block(input_tensor: tf.Tensor, num_filters: int) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation("relu")(x)

    def encoder_block(
        input_tensor: tf.Tensor, num_filters: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = conv_block(input_tensor, num_filters)
        p = tf.keras.layers.MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(
        input_tensor: tf.Tensor, skip_features: tf.Tensor, num_filters: int
    ) -> tf.Tensor:
        x = tf.keras.layers.Conv2DTranspose(
            num_filters, (2, 2), strides=2, padding="same"
        )(input_tensor)
        x = tf.keras.layers.Concatenate()([x, skip_features])
        return conv_block(x, num_filters)

    inputs = tf.keras.layers.Input(shape=[dim, dim, 3])

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return tf.keras.Model(inputs, outputs, name="Vanilla-U-Net")


def build_and_compile_unet(dim: int = DIM, learning_rate: float = 0.001) -> tf.keras.Model:
    """Constrói e compila o modelo U-Net MobileNetV2.

    Args:
        dim: Dimensão das imagens.
        learning_rate: Taxa de aprendizado inicial do otimizador Adam.

    Returns:
        Modelo compilado pronto para treinamento.
    """
    model = unet_model(dim=dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.BinaryIoU(),
            dice_coef,
            iou_coef,
        ],
    )
    return model


def load_unet(model_path: str) -> tf.keras.Model:
    """Carrega um modelo U-Net salvo em disco.

    Args:
        model_path: Caminho para o arquivo .h5 do modelo.

    Returns:
        Modelo Keras carregado com as métricas customizadas registradas.
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"dice_coef": dice_coef, "iou_coef": iou_coef},
    )
