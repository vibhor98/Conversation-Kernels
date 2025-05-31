
"""
Conversation Context Retriever
"""

import torch
import transformers
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from torch import nn

roberta_config = RobertaConfig()

class ConvContextRetriever(RobertaPreTrainedModel):
    def __init__(self, roberta_config, **kwargs):
        super(ConvContextRetriever, self).__init__(roberta_config)

        roberta_config.update(kwargs)
        self.config = roberta_config
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.linear_target = nn.Linear(768, 768)
        self.linear_context = nn.Linear(768, 768)

    def forward(self, sentence_features):
        reps = []
        for indx, sentence_feature in enumerate(sentence_features[:4]):
            roberta_outputs = self.roberta(input_ids=sentence_feature['input_ids'],
                attention_mask=sentence_feature['attention_mask'], token_type_ids=sentence_feature['token_type_ids'])
            reps.append(roberta_outputs.last_hidden_state)

        target_embed = self.linear_target(reps[0][:, 0, :])
        context_embed = [self.linear_context(r[:, 0, :]) for r in reps[1:]]
        context_embed = torch.mean(torch.stack(context_embed), dim=0)

        return torch.Tensor([torch.dot(target_embed[i], context_embed[i]) for i in range(target_embed.shape[0])])
