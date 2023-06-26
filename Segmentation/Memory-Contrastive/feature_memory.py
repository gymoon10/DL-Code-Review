"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import random

class FeatureMemory:
    def __init__(self, num_samples, dataset,  memory_per_class=2048, feature_size=256, n_classes=19):
        self.num_samples = num_samples
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        if dataset == 'cityscapes':  # usually all classes in one image
            self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))  # 3
        elif dataset == 'pascal_voc':  # usually only around 3 classes on each image, except background class
            self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))  # 19


    def add_features_from_sample_learned(self, model, features, class_labels, batch_size):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()  # (M pixels, 256)
        class_labels = class_labels.detach().cpu().numpy()  # (M pixels) M=# of correctly predicted pixels among classes

        elements_per_class = batch_size * self.per_class_samples_per_image  # default: 21

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = (class_labels == c)  # get mask for class c - mask_c.sum() = m pixels

            # selector: Linear -> BN -> LeakyReLU -> Linear(out_dim=1)
            selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
            features_c = features[mask_c, :]  # get features from class c - (m pixels, 256)
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        rank = selector(features_c)  # (m pixels, 1)
                        rank = torch.sigmoid(rank)  # (m pixels, 1)

                        # sort them (ascending order)
                        _, indices = torch.sort(rank[:, 0], dim=0)  # (m pixels, )
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()

                        # get features with the highest rankings
                        features_c = features_c[indices, :]  # (m pixels, 256)
                        new_features = features_c[:elements_per_class, :]  # (elements_per_class, 256)
                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features  # len(memory[c])=elements_per_class

                else:  # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis=0)[:self.memory_per_class, :]



