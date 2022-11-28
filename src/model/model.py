from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel


# 라벨 분포에 따른 weight 생성 함수
def get_label_weight(labels, num_labels):
    number_of_labels = []

    # 입력 데이터의 전체 수를 저장하기위한 변수
    # 데이터 10개 중, 9개가 1번 라벨이고 1개가 2번 라벨일 경우
    # weight는 [0.1, 0.9]로 계산
    number_of_total = torch.zeros(size=(1,), dtype=torch.float, device=torch.device("cuda"))

    for label_index in range(num_labels):
        # 라벨 index를 순차적으로 받아와 현재 라벨(label_index)에 해당하는 데이터 수를 계산
        number_of_label = (labels == label_index).sum(dim=-1).float()

        # 현재 라벨 분포 저장
        number_of_labels.append(number_of_label)

        # 전체 분모에 현재 라벨을 가진 데이터를 합치는 과정
        number_of_total = torch.add(number_of_total, number_of_label).float()

    # 리스트로 선언된 number_of_labels를 torch.tensor() 형태로 변환
    label_weight = torch.stack(tensors=number_of_labels, dim=0)

    # 각 라벨 분포를 전체 데이터 수로 나누어서 라벨 웨이트 계산
    label_weight = torch.ones(size=(1,), dtype=torch.float, device=torch.device("cuda")) - torch.div(label_weight,
                                                                                                     number_of_total)
    return label_weight

class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # electra model의 추가 설정
        # output_attentions : 모든 electra layer(12층)의 attention alignment score
        # output_hidden_states : 모든 electra layer(12층)의 attention output
        # 적용 방법
        # config.output_attentions = True
        # config.output_hidden_states = True

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)

        # final output projection layer(fnn)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # ELECTRA weight 초기화
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]

        # gru_output : [batch_size, seq_length, gru_hidden_size*2]
        gru_output, _ = self.bi_gru(sequence_output)

        # logits : [batch_size, max_length, 2]
        logits = self.qa_outputs(gru_output)

        # start_logits : [batch_size, max_length, 1]
        # end_logits : [batch_size, max_lenght, 1]
        start_logits, end_logits = logits.split(1, dim=-1)

        # start_logits : [batch_size, max_length]
        # end_logits : [batch_size, max_lenght]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # outputs = (start_logits, end_logits)
        outputs = (start_logits, end_logits,) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            # ignored_index : max_length
            ignored_index = start_logits.size(1)

            # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # logg_fct 선언
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 최종 loss 계산
            total_loss = (start_loss + end_loss ) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (total_loss,) + outputs

        return outputs # (loss), start_logits, end_logits, sent_token_logits
