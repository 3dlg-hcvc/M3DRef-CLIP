import lightning.pytorch as pl
import torch.nn as nn
import torch


class SinglePairContrastiveLoss(pl.LightningModule):
    def __init__(self, temperature, split_batch):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(temperature, device=self.device, dtype=torch.float32))
        self.split_batch = split_batch

    def forward(self, aabb_features, sentence_features, gt_labels, multiple_gt=False):
        gt_label_indices = torch.where(gt_labels)

        aabb_features_filtered = aabb_features[gt_label_indices[0], gt_label_indices[2]]
        sentence_features_filtered = sentence_features.reshape(
            gt_labels.shape[0], gt_labels.shape[1], -1
        )[gt_label_indices[0], gt_label_indices[1]]

        # # gt_labels_flatten = torch.empty(size=)
        # if not gt_mask.any():
        #     return 0.0
        #
        # if not multiple_gt:
        #     aabb_features_filtered = torch.einsum("abc,adb->adc", aabb_features, gt_labels)[gt_mask].flatten(0, 1)
        # else:
        #     # aabb_features_filtered = torch.empty(
        #     #     size=(gt_count.sum(), aabb_features.shape[-1]), dtype=aabb_features.dtype, device=self.device
        #     # )
        #
        #
        #     aabb_features_filtered = aabb_features[gt_indices].flatten(0, 1)
        #
        #
        #     raise NotImplementedError

        # batch_size, lang_chunk_size = aabb_features_filtered.shape[0:2]

        # TODO: refactor code

        # if self.split_batch:
        #     sentence_features_batch = sentence_features.reshape(batch_size, lang_chunk_size, -1)
        #     total_loss = 0
        #     for batch_i in range(batch_size):
        #         # normalized features
        #         normalized_aabb_features = nn.functional.normalize(aabb_features_filtered.reshape(batch_size, lang_chunk_size, -1)[batch_i], dim=1)
        #         normalized_sentence_features = nn.functional.normalize(sentence_features_batch[batch_i], dim=1)
        #
        #         logit_scale = self.logit_scale.exp()
        #         logits_1 = logit_scale * normalized_aabb_features @ normalized_sentence_features.t()
        #         logits_2 = logit_scale * normalized_sentence_features @ normalized_aabb_features.t()
        #
        #         labels = torch.arange(lang_chunk_size, device=self.device, dtype=torch.uint8)  # max 255
        #         loss_a = nn.functional.cross_entropy(logits_1, labels)
        #         loss_b = nn.functional.cross_entropy(logits_2, labels)
        #         total_loss += (loss_a + loss_b) / 2
        #
        #     return total_loss / batch_size
        # else:

        # normalized features
        normalized_aabb_features = nn.functional.normalize(aabb_features_filtered, dim=1)
        normalized_sentence_features = nn.functional.normalize(sentence_features_filtered, dim=1)

        logit_scale = self.logit_scale.exp()
        logits_1 = logit_scale * normalized_aabb_features @ normalized_sentence_features.t()
        logits_2 = logit_scale * normalized_sentence_features @ normalized_aabb_features.t()

        labels = torch.arange(normalized_aabb_features.shape[0], device=self.device, dtype=torch.uint8)  # max 255
        loss_a = nn.functional.cross_entropy(logits_1, labels)
        loss_b = nn.functional.cross_entropy(logits_2, labels)
        return (loss_a + loss_b) / 2
