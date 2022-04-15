import torch.nn as nn
from transformers import T5ForConditionalGeneration

# Defining the pytorch model

class LaTr_for_pretraining(nn.Module):
    def __init__(self, config):

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
        total_feat = self.classification_layer(total_feat)

        if predict_proba:
          return total_feat.softmax(axis = -1)
        
        if predict_class:
          return total_feat.argmax(axis = -1)

        return total_feat