from skimage.color.adapt_rgb import adapt_rgb, each_channel


@adapt_rgb(each_channel)
def apply_filter_each_chanel(image, fn):
    return fn(image)


@adapt_rgb(each_channel)
def merge_image_mask(image, mask):
    return image * mask