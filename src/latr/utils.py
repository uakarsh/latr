import random
import torch

def find_pad_idx(boxes):
  for i, j in enumerate(boxes):
    if int(boxes[i].sum().item()) == 0:
      return i
  return i



def apply_mask_on_token_bbox(boxes, tokenized_words, only_actual_words = False, span = 4, proportion_to_mask = 0.15, special_token = 103):
  
  '''
  code taken from here: https://www.geeksforgeeks.org/python-non-overlapping-random-ranges/

  Note: A more robust solution is to be coded 
  '''
  length_to_be_masked = int(proportion_to_mask*len(boxes))

  if only_actual_words:
    tot = find_pad_idx(tokenized_words)
  else:
    tot = len(boxes)
  
  res = set()
  for _ in range(length_to_be_masked):
    temp = random.randint(0, tot - span) 
    while any(((temp >= idx) and (temp <= idx + span)) for idx in res):
      temp = random.randint(0, tot - span) 
    res.add(temp)

    ## Applying the mask on token
    tokenized_words[temp] = special_token

    ## Applying the masking on the box
    boxes[temp, 0] = torch.min(boxes[temp: temp+span, 0])
    boxes[temp, 1] = torch.min(boxes[temp: temp+span, 1])
    boxes[temp, 2] = torch.max(boxes[temp: temp+span, 2])
    boxes[temp, 3] = torch.max(boxes[temp: temp+span, 3])
    boxes[temp, 4] = boxes[temp, 2] - boxes[temp, 0]
    boxes[temp, 5] = boxes[temp, 3] - boxes[temp, 1]

  return res,boxes, tokenized_words