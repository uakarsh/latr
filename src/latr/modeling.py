import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration, ViTModel

# Defining the pytorch model

class LaTr_for_pretraining(nn.Module):
    def __init__(self, config, classify = False):

      super(LaTr_for_pretraining, self).__init__()
      self.vocab_size = config['vocab_size']

      model = T5ForConditionalGeneration.from_pretrained(config['t5_model'])
      dummy_encoder = list(nn.Sequential(*list(model.encoder.children())[1:]).children())   ## Removing the Embedding layer
      dummy_decoder = list(nn.Sequential(*list(model.decoder.children())[1:]).children())   ## Removing the Embedding Layer

      ## Using the T5 Encoder

      self.list_encoder = nn.Sequential(*list(dummy_encoder[0]))
      self.residue_encoder = nn.Sequential(*list(dummy_encoder[1:]))
      self.list_decoder = nn.Sequential(*list(dummy_decoder[0]))
      self.residue_decoder = nn.Sequential(*list(dummy_decoder[1:]))

      self.language_emb = nn.Embedding(config['vocab_size'], config['hidden_state'])

      self.top_left_x = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.bottom_right_x = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.top_left_y = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.bottom_right_y = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.width_emb = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.height_emb = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])

      self.classify = classify
      self.classification_layer = nn.Linear(config['hidden_state'], config['classes'])

    def forward(self, tokens, coordinates, predict_proba = False, predict_class = False):

        batch_size = len(tokens)
        embeded_feature = self.language_emb(tokens)
        
        top_left_x_feat =     self.top_left_x(coordinates[:,:, 0])
        top_left_y_feat =     self.top_left_y(coordinates[:,:, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:,:, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:,:, 3])
        width_feat =          self.width_emb(coordinates[:,:, 4])
        height_feat =         self.height_emb(coordinates[:,:, 5])

        total_feat = embeded_feature + top_left_x_feat + top_left_y_feat + bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat

        ## Extracting the feature

        for layer in self.list_encoder:
          total_feat = layer(total_feat)[0]
        total_feat = self.residue_encoder(total_feat)

        for layer in self.list_decoder:
          total_feat = layer(total_feat)[0]
        total_feat = self.residue_decoder(total_feat)

        if  self.classify :
          total_feat = self.classification_layer(total_feat)

        if predict_proba:
          return total_feat.softmax(axis = -1)
        
        if predict_class:
          return total_feat.argmax(axis = -1)

        return total_feat


class LaTr_for_finetuning(nn.Module):
  def __init__(self, config, address_to_pre_trained_weights = None):
    super(LaTr_for_finetuning, self).__init__()

    self.config = config
    self.vocab_size = config['vocab_size']
    self.question_emb = nn.Embedding(config['vocab_size'], config['hidden_state'])

    self.pre_training_model = LaTr_for_pretraining(config)
    if address_to_pre_trained_weights is not None:
      self.pre_training_model.load_state_dict(torch.load(address_to_pre_trained_weights))
    self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
  
    ## In the fine-tuning stage of vit, except the last layer, all the layers were freezed

    self.classification_head = nn.Linear(config['hidden_state'], config['vocab_size'])

  def forward(self, lang_vect, spatial_vect, quest_vect, img_vect):

    
    ## The below block of code calculates the language and spatial featuer
    embeded_feature =     self.pre_training_model.language_emb(lang_vect)
    top_left_x_feat =     self.pre_training_model.top_left_x(spatial_vect[:,:, 0])
    top_left_y_feat =     self.pre_training_model.top_left_y(spatial_vect[:,:, 1])
    bottom_right_x_feat = self.pre_training_model.bottom_right_x(spatial_vect[:,:, 2])
    bottom_right_y_feat = self.pre_training_model.bottom_right_y(spatial_vect[:,:, 3])
    width_feat =          self.pre_training_model.width_emb(spatial_vect[:,:, 4])
    height_feat =         self.pre_training_model.height_emb(spatial_vect[:,:, 5])

    spatial_lang_feat = embeded_feature + top_left_x_feat + top_left_y_feat + bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat


    ## Extracting the image feature, using the Vision Transformer
    img_feat = self.vit(img_vect).last_hidden_state
    
    ## Extracting the question vector
    quest_feat = self.question_emb(quest_vect)

    ## Concating the three features, and then passing it through the T5 Transformer
    final_feat = torch.cat([img_feat, spatial_lang_feat,quest_feat ], axis = -2)

    ## Passing through the T5 Transformer
    for layer in self.pre_training_model.list_encoder:
        final_feat = layer(final_feat)[0]

    final_feat = self.pre_training_model.residue_encoder(final_feat)

    for layer in self.pre_training_model.list_decoder:
        final_feat = layer(final_feat)[0]
    final_feat = self.pre_training_model.residue_decoder(final_feat)

    answer_vector = self.classification_head(final_feat)[:, :self.config['seq_len'], :]

    return answer_vector
