import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    PreTrainedModel,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class QAModel(PreTrainedModel):
    """
    Based on BertForQuestionAnswering but considers last n hidden layers
    in the prediction head, concatenating or summing them.

    Parameters:
      config.cat_n: number of last hidden layers to consider
      config.cat_op: 'sum' or 'cat'
    """

    def __init__(self, config, pretrained_backbone=None):
        super().__init__(config=config)
        if pretrained_backbone:
            self.backbone = AutoModel.from_pretrained(pretrained_backbone, config=config)
        else:
            self.backbone = AutoModel.from_config(config)
        self.num_labels = config.num_labels
        mul = config.cat_n if config.cat_op == 'cat' else 1
        self.qa_outputs = nn.Linear(mul * config.hidden_size, config.num_labels)

    def forward(self, **kwargs):
        start_positions = kwargs.pop('start_positions', None)
        end_positions = kwargs.pop('end_positions', None)

        kwargs['output_hidden_states'] = True
        outputs = self.backbone(**kwargs)

        if self.config.cat_op == 'cat':
            # len = 13
            # pos 0: position embedding
            # [?, (bs, seq, hidden), ..., (bs, seq, hidden)]
            cat_output = torch.cat(outputs.hidden_states[-self.config.cat_n:], dim=2)
        elif self.config.cat_op == 'sum':
            cat_output = sum(outputs.hidden_states[-self.config.cat_n:])

        # (bs, seq, hidden) -> (bs, seq, num_labels = 2)
        logits = self.qa_outputs(cat_output)
        #print(logits.size())
        start_logits, end_logits = logits.split(1, dim=-1) # (size of single chunk, dim)
        start_logits = start_logits.squeeze(-1).contiguous() # (bs, seq, 1) -> (bs, seq)
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1: 
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            # clamps all elements in input into the range [min, max]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_qa_model(pretrained, cat_n=1, cat_op='cat'):
    config = AutoConfig.from_pretrained(pretrained)
    if config.architectures == ["QAModel"]:
        return load_qa_model(pretrained)
    config.cat_n = cat_n
    config.cat_op = cat_op
    config.num_labels = 2
    if cat_n == 1:
        return AutoModelForQuestionAnswering.from_pretrained(pretrained, config=config)
    return QAModel(config, pretrained_backbone=pretrained)


def load_qa_model(path):
    config = AutoConfig.from_pretrained(path)
    if config.cat_n == 1:
        return AutoModelForQuestionAnswering.from_pretrained(path, config=config)
    return QAModel.from_pretrained(path, config=config)


def load_swag_model(path):
    config = AutoConfig.from_pretrained(path)
    return AutoModelForMultipleChoice.from_pretrained(path, config=config)
