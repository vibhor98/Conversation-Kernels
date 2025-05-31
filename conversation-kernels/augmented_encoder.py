
"""
Context-Augmented Encoder
"""

import torch

from torch import nn
from context_retriever import ConvContextRetriever
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaForSequenceClassification, RobertaPreTrainedModel, RobertaPooler

roberta_config = RobertaConfig()

class ContextAugmentedEncoder(RobertaPreTrainedModel):
    def __init__(self, roberta_config, **kwargs):
        super(ContextAugmentedEncoder, self).__init__(roberta_config)

        roberta_config.classifier_dropout = roberta_config.hidden_dropout_prob
        roberta_config.update(kwargs)
        roberta_config.num_labels = 2
        self.config = roberta_config
        self.roberta_encoder = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

        self.context_retriever = ConvContextRetriever.from_pretrained('roberta-base')
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence_features):
        output_dot_prod = self.context_retriever(sentence_features)
        p_zx = self.softmax(output_dot_prod)    # P(z | x)

        p_y_zx = self.roberta_encoder(input_ids=sentence_features[-1]['input_ids'],
            attention_mask=sentence_features[-1]['attention_mask'], token_type_ids=sentence_features[-1]['token_type_ids'])   # P(y | z, x)

        return torch.matmul(p_zx.to('cuda:1'), p_y_zx.logits.softmax(dim=-1))
