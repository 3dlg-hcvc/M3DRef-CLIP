import torch.nn as nn
import lightning.pytorch as pl
from torch.nn.utils.rnn import pad_packed_sequence
from m3drefclip.model.cross_modal_module.attention import MultiHeadAttention
import torch


class GRUTextEncoder(pl.LightningModule):
    def __init__(self, embedding_dim, gru_hidden_size, output_channel):
        super().__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_hidden_size, batch_first=True, bidirectional=False)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_size, output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.LayerNorm(normalized_shape=output_channel)
        )
        self.attention = MultiHeadAttention(d_model=output_channel, h=4, d_k=16, d_v=16, dropout=0.1)
        self.text_projection = nn.Parameter(
            torch.empty(
                size=(gru_hidden_size, output_channel),
                device=self.device,
                dtype=torch.float32
            )
        )
        self._weight_initialization(output_channel)

    def _weight_initialization(self, output_channel):
        nn.init.normal_(self.text_projection, std=output_channel**-0.5)

    def forward(self, data_dict, output_dict):

        word_embeddings = data_dict["word_embeddings"]
        # lang_len = data_dict["lang_len_list"]
        # batch_size, len_nun_max, max_des_len = word_embeddings.shape[:3]
        #
        # word_embeddings = word_embeddings.reshape(batch_size * len_nun_max, max_des_len, -1)
        # lang_len = lang_len.reshape(batch_size * len_nun_max)
        # first_obj = data_dict["first_obj_list"].reshape(batch_size * len_nun_max)

        # if data_dict["istrain"][0] == 1 and random.random() < 0.5:
        #     for i in range(word_embeddings.shape[0]):
        #         word_embeddings[i, first_obj] = data_dict["unk"][0]
        #         len = lang_len[i]
        #         for j in range(int(len / 5)):
        #             num = random.randint(0, len - 1)
        #             word_embeddings[i, num] = data_dict["unk"][0]
        # elif data_dict["istrain"][0] == 1:
        #     for i in range(word_embeddings.shape[0]):
        #         len = lang_len[i]
        #         for j in range(int(len / 5)):
        #             num = random.randint(0, len - 1)
        #             word_embeddings[i, num] = data_dict["unk"][0]
        #
        # # Reverse
        # main_lang_len = data_dict["main_lang_len_list"]
        # main_lang_len = main_lang_len.reshape(batch_size * len_nun_max)
        #
        # if data_dict["istrain"][0] == 1 and random.random() < 0.5:
        #     for i in range(word_embeddings.shape[0]):
        #         new_word_emb = copy.deepcopy(word_embeddings[i])
        #         new_len = lang_len[i] - main_lang_len[i]
        #         new_word_emb[:new_len] = word_embeddings[i, main_lang_len[i]:lang_len[i]]
        #         new_word_emb[new_len:lang_len[i]] = word_embeddings[i, :main_lang_len[i]]
        #         word_embeddings[i] = new_word_emb

        out, h_n = self.gru(word_embeddings)
        padded_features, features_lens = pad_packed_sequence(out, batch_first=True)
        padded_features = self.mlp(padded_features)
        output_dict["word_features"] = self.attention(
            padded_features, padded_features, padded_features, attention_mask=data_dict["lang_attention_mask"]
        )
        output_dict["sentence_features"] = h_n.permute(1, 0, 2).contiguous().flatten(start_dim=1) @ self.text_projection

        # if self.use_sem_classifier:
        #     sentence_features = h_n.permute(1, 0, 2).contiguous().flatten(start_dim=1)
        #     output_dict["pred_lang_sem_label"] = self.sem_classifier(sentence_features)
