import copy
import torch


def pad_sequence(sequence, pad_value):
    '''
    A function to pad a sequence of tensors to the maximum length tensor, currently it supports 1d and 2d tensors
    Arguments:
        sequence: A list of tensors
        pad_value: The value to pad the tensors with
    Returns:
        A tensor with the padded tensors
    '''
    max_len = 0
    for i in sequence:
        max_len = max(max_len, len(i))

    for i, _ in enumerate(sequence):
        pad_length = max_len - len(_)
        if pad_length != 0:
            pad_entry = torch.stack([pad_value for j in range(pad_length)])
            sequence[i] = torch.cat([sequence[i], pad_entry])

    return torch.stack(sequence)


def collate(data_bunch):
    '''
    A function to collate the data bunch
    Arguments:
        data_bunch: A list of dictionaries containing the data
    Returns:
        A dictionary containing the collated data
    '''
    dict_data_bunch = {}

    for i in data_bunch:
        for (key, value) in i.items():
            if key not in dict_data_bunch:
                dict_data_bunch[key] = []
            dict_data_bunch[key].append(value)

    images = torch.stack(dict_data_bunch.pop('pixel_values'), axis=0)
    for entry in dict_data_bunch:
        if entry == "bbox":
            dict_data_bunch[entry] = pad_sequence(
                dict_data_bunch[entry], torch.as_tensor([0, 0, 0, 0, 0, 0]))
        elif entry == 'labels':
            dict_data_bunch[entry] = pad_sequence(
                dict_data_bunch[entry], torch.as_tensor(-100))
        else:
            dict_data_bunch[entry] = pad_sequence(
                dict_data_bunch[entry], torch.as_tensor(0))
    return {"img": images, **dict_data_bunch}


def draw_bounding_box_on_pil_image(img, bounding_box, outline='violet'):
    '''
    A function to draw bounding boxes on PIL images
    Arguments:
        img: A PIL image
        bounding_box: A list containing the bounding box coordinates
    Returns:
        A PIL image with the bounding box drawn
    '''
    from PIL import ImageDraw
    img = copy.deepcopy(img)
    if img.size != (1000, 1000):
        img = img.resize((1000, 1000))

    draw = ImageDraw.Draw(img)
    for box in bounding_box:
        if type(box) == torch.Tensor:
            box = box.tolist()
        draw.rectangle(box[:4], outline=outline)
    return img
