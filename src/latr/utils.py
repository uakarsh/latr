import random
import torch
import math
from torch.nn.utils.rnn import pad_sequence


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


def convert_ans_to_token(answer, label2id, max_seq_length = 512 ):

  ## Simple Trick to pad a sequence to deired length
  dummy_array = torch.zeros(max_seq_length)
  actual_ans_array = []

  answer = answer.split(" ")
  for token in answer:
    actual_ans_array.append(label2id[token]['id'])
  
  actual_ans_array = torch.tensor(actual_ans_array, dtype = torch.int32)
  actual_ans_array = pad_sequence([actual_ans_array,dummy_array], batch_first  = True)[0]

  return actual_ans_array


def convert_ques_to_token(question, tokenizer, pad_token_id = 0, max_seq_len = 512):

  question_array = []
  question = question.split(" ")
  
  for token in question:
    question_array.extend(tokenizer(token, add_special_tokens = False).input_ids)
  
  if len(question_array)< max_seq_len:
        question_array.extend([pad_token_id]* (max_seq_len-len(question_array)))

  question_array = torch.tensor(question_array, dtype = torch.int32)
  return question_array[:max_seq_len]


## To be taken from here
## https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    
    modified from answer here: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    # angle = np.deg2rad(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def convert_token_to_ques(ques, tokenizer):
  decoded_ques = tokenizer.decode(ques, skip_special_tokens=True)
  return decoded_ques


def convert_token_to_answer(ans, id2label):
  non_zero_argument = torch.nonzero(ans,as_tuple = False).view(-1)

  actual_answer = ans[non_zero_argument].cpu().numpy()
  decoded_answer = []
  
  for token in actual_answer:
    decoded_answer.append(id2label[token])
  
  decoded_answer = " ".join(decoded_answer)
  return decoded_answer

