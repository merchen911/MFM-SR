# -*- coding: utf-8 -*-
# @Time    : 2022/02/22 19:32
# @Author  : Peilin Zhou, Yueqi Xie
# @Email   : zhoupl@pku.edu.cn
r"""
SASRecD
################################################

Reference:
    Yueqi Xie and Peilin Zhou et al. "Decouple Side Information Fusion for Sequential Recommendation"
    Submited to SIGIR 2022.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer,DIFTransformerEncoder
from recbole.model.loss import BPRLoss
import copy

import warnings
warnings.filterwarnings(action='ignore')


class SASRecDM_MTL(SequentialRecommender):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, dataset):
        super(SASRecDM_MTL, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_features = config['selected_features']
        self.tabular_feature_names = list(self.selected_features)
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']
        # 멀티모달 합성(MFM): DIF의 fusion_type(어텐션 스코어 합성)과 별개. 키는 tabular 필드명 + 'image'/'text'
        # RecBole Config: 없는 키는 config['k']가 None → 기본값 처리 (config.get 사용 불가)
        _mfm_fusion = config['MFM_fusion']
        self.MFM_fusion = _mfm_fusion if _mfm_fusion is not None else 'sum'
        self._mfm_dim = self.attribute_hidden_size[0]
        default_mfm = list(self.tabular_feature_names) + ['image', 'text']
        _mfm_modalities = config['mfm_modalities']
        self.mfm_modalities = tuple(_mfm_modalities) if _mfm_modalities is not None else tuple(default_mfm)
        self.mfm_modalities_set = frozenset(self.mfm_modalities)
        allowed_mfm = frozenset(self.tabular_feature_names) | {'image', 'text'}
        for m in self.mfm_modalities:
            if m not in allowed_mfm:
                raise ValueError(
                    f"mfm_modalities entry {m!r} must be in tabular {self.tabular_feature_names!r} or 'image'/'text'"
                )
        if len(self.mfm_modalities) < 1:
            raise ValueError('mfm_modalities must contain at least one modality')
        for m in self.mfm_modalities:
            if m in self.tabular_feature_names:
                i = self.tabular_feature_names.index(m)
                if self.attribute_hidden_size[i] != self._mfm_dim:
                    raise ValueError(
                        f'MFM expects embedding dim {self._mfm_dim} for all branches; '
                        f'{m!r} uses attribute_hidden_size[{i}]={self.attribute_hidden_size[i]}'
                    )
        self._mfm_n = len(self.mfm_modalities)
        if self.MFM_fusion == 'gate':
            self.mfm_gates = nn.ParameterDict({
                m: nn.Parameter(torch.ones(1, 1, self._mfm_dim)) for m in self.mfm_modalities
            })
        elif self.MFM_fusion == 'concat':
            self.mfm_concat_proj = nn.Linear(self._mfm_dim * self._mfm_n, self._mfm_dim)
        elif self.MFM_fusion != 'sum':
            raise ValueError(f"MFM_fusion must be 'sum', 'gate', or 'concat', got {self.MFM_fusion!r}")

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        # *add multimodal feature
        pretrained_item_emb_image = dataset.get_preload_weight('itemid_image')
        pretrained_item_emb_text = dataset.get_preload_weight('itemid_text')
        self.item_embedding_image = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb_image))
        self.item_embedding_text = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb_text))
        self.image_linear = torch.nn.Linear((4096), self.attribute_hidden_size[0])
        self.text_linear = torch.nn.Linear((768), self.attribute_hidden_size[0])
        
        self.feature_embed_layer_list = nn.ModuleList(
            [copy.deepcopy(FeatureSeqEmbLayer(dataset,self.attribute_hidden_size[_],[self.tabular_feature_names[_]],self.pooling_mode,self.device)) for _
             in range(len(self.tabular_feature_names))])

        self.trm_encoder = DIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=[self._mfm_dim],
            feat_num=1,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        self.n_attributes = {}
        for attribute in self.selected_features:
            self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.hidden_size),
                                             nn.BatchNorm1d(num_features=self.hidden_size),
                                             nn.ReLU(),
                                             # final logits
                                             nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.n_attributes)
                                             )
        elif self.attribute_predictor == 'linear':
            self.ap = nn.ModuleList(
                [copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_]))
                 for _ in self.selected_features])
            self.image_predictor = nn.Linear(in_features=self.hidden_size, out_features=4096) 
            self.text_predictor = nn.Linear(in_features=self.hidden_size, out_features=768) 

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            self.img_text_loss_fct = nn.MSELoss() # add reconstuction loss 
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        self.selected_features.append('image')
        # self.num_feature_field += 1

        self.selected_features.append('text')
        self.num_feature_field += 1
        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def _merge_tabular_sparse_dense(sparse_embedding, dense_embedding):
        if sparse_embedding is None and dense_embedding is None:
            return None
        if sparse_embedding is None:
            return dense_embedding
        if dense_embedding is None:
            return sparse_embedding
        return sparse_embedding + dense_embedding

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        feature_table = {}
        for idx, feature_embed_layer in enumerate(self.feature_embed_layer_list):
            name = self.tabular_feature_names[idx]
            if name not in self.mfm_modalities_set:
                continue
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding['item']
            dense_embedding = dense_embedding['item']
            merged = self._merge_tabular_sparse_dense(sparse_embedding, dense_embedding)
            if merged is not None:
                feature_table[name] = merged

        if 'image' in self.mfm_modalities_set:
            feature_table['image'] = self.image_linear(
                self.item_embedding_image(item_seq).unsqueeze(-2).float())
        if 'text' in self.mfm_modalities_set:
            feature_table['text'] = self.text_linear(
                self.item_embedding_text(item_seq).unsqueeze(-2).float())

        mfm_tensors = []
        for m in self.mfm_modalities:
            if m not in feature_table:
                raise KeyError(
                    f"MFM modality {m!r} missing from feature_table (no embedding produced). "
                    f'Keys present: {sorted(feature_table.keys())}'
                )
            mfm_tensors.append(feature_table[m])

        # MFM_fusion: 'sum', 'gate', 'concat'
        if self.MFM_fusion == 'sum':
            fused = mfm_tensors[0]
            for t in mfm_tensors[1:]:
                fused = fused + t
            feature_emb = [fused]
        elif self.MFM_fusion == 'concat':
            fused = torch.cat(mfm_tensors, dim=-1)
            feature_emb = [self.mfm_concat_proj(fused)]
        elif self.MFM_fusion == 'gate':
            gated = None
            for m, t in zip(self.mfm_modalities, mfm_tensors):
                g = torch.sigmoid(self.mfm_gates[m] * t)
                gt = t * g
                gated = gt if gated is None else gated + gt
            feature_emb = [gated]
        else:
            raise ValueError(f'Unknown MFM_fusion type: {self.MFM_fusion}')
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, feature_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1) #아이템 시퀀스 별 맨 마지막 인덱스 값 가져오기
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # batch * item_num
            loss = self.loss_fct(logits, pos_items) # logits(각 아이템별 스코어)와 실제 라벨과 CE loss 계산
            if self.attribute_predictor!='' and self.attribute_predictor!='not':
                loss_dic = {'item_loss':loss}
                attribute_loss_sum = 0
                for i, a_predictor in enumerate(self.ap):
                    attribute_logits = a_predictor(seq_output) # batch * categories_num
                    attribute_labels = interaction.interaction[self.selected_features[i]]
                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes[
                        self.selected_features[i]])

                    if len(attribute_labels.shape) > 2:
                        attribute_labels = attribute_labels.sum(dim=1)
                    attribute_labels = attribute_labels.float()
                    attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                    attribute_loss = torch.mean(attribute_loss[:, 1:]) # 0번 인덱스는 패딩이므로 제외
                    loss_dic[self.selected_features[i]] = attribute_loss

                    image_logits = self.image_predictor(seq_output).unsqueeze(-2) # batch * 1 * 4096
                    image_labels = self.item_embedding_image(item_seq).float() # batch * item_seq_len * 4096
                    image_loss = self.img_text_loss_fct(image_logits, image_labels) # broad casting 
                    loss_dic['image'] = image_loss
                    
                    text_logits = self.text_predictor(seq_output).unsqueeze(-2) # batch * 1 * 768
                    text_labels = self.item_embedding_text(item_seq).float() # batch * item_seq_len * 768
                    text_loss = self.img_text_loss_fct(text_logits, text_labels) # broad casting 
                    loss_dic['text'] = text_loss

                if self.num_feature_field == 1:
                    total_loss = loss + self.lamdas[0] * attribute_loss
                    # print('total_loss:{}\titem_loss:{}\tattribute_{}_loss:{}'.format(total_loss, loss,self.selected_features[0],attribute_loss))
                else:
                    for i,attribute in enumerate(self.selected_features):
                        attribute_loss_sum += self.lamdas[i] * loss_dic[attribute]
                    total_loss = loss + attribute_loss_sum
                    loss_dic['total_loss'] = total_loss
#                     s = ''
#                     for key,value in loss_dic.items():
#                         s += '{}_{:.4f}\t'.format(key,value.item())
#                     print(s)
            else:
                total_loss = loss
            return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores