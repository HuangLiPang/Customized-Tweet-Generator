import json

from commons import get_model, transform_image, generate_caption

model = get_model()
imagenet_class_index = json.load(open('imagenet_class_index.json'))


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = generate_caption(model, tensor)
    except Exception as err:
        print(str(err))
        return 0, 'error'
    return outputs
