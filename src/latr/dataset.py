import os
import json
import numpy as np
from google.colab.patches import cv2_imshow
import pytesseract
from PIL import Image, ImageDraw

PAD_TOKEN_BOX = [0, 0, 0, 0]
max_seq_len = 512

## Function: 1
## Purpose: Resize and align the bounding box for the different sized image

def resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h):
    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    orig_left, orig_top, orig_right, orig_bottom = bbox
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    xmax = int(np.round(orig_right * x_scale))
    ymax = int(np.round(orig_bottom * y_scale))
    return [x, y, xmax, ymax]

## Function: 2
## Purpose: Reading the json file from the path and return the dictionary

def load_json_file(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

## Function: 3
## Purpose: Getting the address of specific file type, eg: .pdf, .tif, so and so

def get_specific_file(path, last_entry = 'tif'):
  base_path = path
  for i in os.listdir(path):
    if i.endswith(last_entry):
      return os.path.join(base_path, i)

  return '-1'


## Function: 4


def get_tokens_with_boxes(unnormalized_word_boxes, list_of_words, tokenizer, pad_token_id = 0, pad_token_box = [0, 0, 0, 0], max_seq_len = 512):
    
    '''
    This function returns two items:
    1. unnormalized_token_boxes -> a list of len = max_seq_len, containing the boxes corresponding to the tokenized words, 
                                    one box might repeat as per the tokenization procedure
    2. tokenized_words -> tokenized words corresponding to the tokenizer and the list_of_words
    '''

    assert len(unnormalized_word_boxes) == len(list_of_words), "Bounding box length!= total words length"
    
    length_of_box = len(unnormalized_word_boxes)
    unnormalized_token_boxes = []
    tokenized_words = []

    for box, word in zip(unnormalized_word_boxes, list_of_words):
      current_tokens = tokenizer(word, add_special_tokens = False).input_ids
      unnormalized_token_boxes.extend([box]*len(current_tokens))
      tokenized_words.extend(current_tokens)

    if len(unnormalized_token_boxes)<max_seq_len:
        unnormalized_token_boxes.extend([pad_token_box] * (max_seq_len-len(unnormalized_token_boxes)))
        
    if len(tokenized_words)< max_seq_len:
        tokenized_words.extend([pad_token_id]* (max_seq_len-len(tokenized_words)))
        
    return unnormalized_token_boxes[:max_seq_len], tokenized_words[:max_seq_len]

## Function: 5
## Function, which would only be used when the below function is used

def get_topleft_bottomright_coordinates(df_row):
    left, top, width, height = df_row["left"], df_row["top"], df_row["width"], df_row["height"]
    return [left, top, left + width, top + height]

## Function: 6
## If the OCR is not provided, this function would help in extracting OCR


def apply_ocr(image_fp):
    """
    Returns words and its bounding boxes from an image
    """
    img = Image.open(tif_path).convert("RGB")
    img = img.resize(resize_shape)
    width, height = img.size

    ocr_df = pytesseract.image_to_data(img, output_type="data.frame")
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text.apply(lambda x: str(x).strip()))
    actual_bboxes = ocr_df.apply(get_topleft_bottomright_coordinates, axis=1).values.tolist()

    ## Normalizing the bounding box
    actual_bboxes = list(map(lambda x: resize_align_bbox(x, width, height, 1,1), actual_bboxes))

    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}

## Function: 6
## If the OCR is not provided, this function would help in extracting OCR


def apply_ocr(image_fp):
    """
    Returns words and its bounding boxes from an image
    """
    img = Image.open(tif_path).convert("RGB")
    img = img.resize(resize_shape)
    width, height = img.size

    ocr_df = pytesseract.image_to_data(img, output_type="data.frame")
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text.apply(lambda x: str(x).strip()))
    actual_bboxes = ocr_df.apply(get_topleft_bottomright_coordinates, axis=1).values.tolist()

    ## Normalizing the bounding box
    actual_bboxes = list(map(lambda x: resize_align_bbox(x, width, height, 1,1), actual_bboxes))

    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}

## Function: 7
## Merging all the above functions, for the purpose of extracting the image, bounding box and the tokens (sentence wise)


def create_features(
    image_path,
    tokenizer,
    target_size = (1000, 1000),
    max_seq_length=512,
    use_ocr = False,
    bounding_box = None,
    words = None
    ):
  
  '''
  We assume that the bounding box provided are normalized, so that we just need to scale it up
  '''


  img = Image.open(image_path).convert("RGB")
  img = img.resize(target_size)
  width, height = img.size
  
  if (use_ocr == False) and (bounding_box == None or words == None):
    raise Exception('Please provide the bounding box and words or pass the argument "use_ocr" = True')
  
  if use_ocr == True:
    entries = apply_ocr(img)
    bounding_box = entries["bbox"]
    words = entries["words"]
  
  boxes, tokenized_words = get_tokens_with_boxes(unnormalized_word_boxes = bounding_box,
                                               list_of_words = words, 
                                               tokenizer = tokenizer,
                                               pad_token_id = 0,
                                               pad_token_box = PAD_TOKEN_BOX,
                                               max_seq_len = max_seq_len
                                               )

  return img, boxes, tokenized_words