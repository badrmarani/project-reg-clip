import torch
from torch import nn
from PIL import Image
from skimage import io
import os
import clip

from .utils import extract_keywords, is_keyword_in_caption


def nonzero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(tensor).squeeze(dim=-1)


def _node_get(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


class CLIPLoss(nn.Module):
    def __init__(
        self,
        dataset_root_dir: str,
        normalize: bool = True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.dataset_root_dir = dataset_root_dir
        self.normalize = normalize
        self.device = device

        torch._C.Node.__getitem__ = _node_get
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=True
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def clip_score(self, filenames, keywords):
        filenames = [os.path.join(self.dataset_root_dir, f) for f in filenames]
        images = torch.cat(
            [self.preprocess(Image.open(f)).unsqueeze(0) for f in filenames]
        ).to(self.device)
        images_features = self.model.encode_image(images)

        k = torch.cat([clip.tokenize(f"a photo of a {k}") for k in keywords]).to(
            self.device
        )
        texts_features = self.model.encode_text(k)

        images_features /= images_features.norm(dim=-1, keepdim=True)
        texts_features /= texts_features.norm(dim=-1, keepdim=True)
        similarity = 100.0 * images_features @ texts_features.T  # (bs, 20)
        return similarity.mean(dim=0)

    def forward_similarity(self, filenames_wrong, filenames_correct, keywords):
        sim_wrong = self.clip_score(filenames_wrong, keywords)
        sim_correct = self.clip_score(filenames_correct, keywords)
        sim = sim_correct - sim_wrong

        if self.normalize:
            sim -= sim.min()
            sim /= sim.max()
        return sim

    def forward(self, y_pred, y_true, groups, filenames, captions, loss):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size)
        y_true = y_true.view(batch_size)
        groups = groups.view(batch_size)

        idx = torch.arange(0, batch_size, device=self.device)
        a = torch.stack((y_true, y_pred, groups, idx), dim=1)

        is_correct = a[:, 0] == a[:, 1]

        correct = a[nonzero(is_correct == 1), :]
        correct_class_0 = correct[nonzero(correct[:, 0] == 0), :]
        correct_class_1 = correct[nonzero(correct[:, 0] == 1), :]

        wrong = a[nonzero(is_correct == 0), :]
        wrong_class_0 = wrong[nonzero(wrong[:, 0] == 0), :]
        wrong_class_1 = wrong[nonzero(wrong[:, 0] == 1), :]

        wrong_class_0_filenames = [
            f for i, f in enumerate(filenames) if i in wrong_class_0[:, -1]
        ]
        correct_class_0_filenames = [
            f for i, f in enumerate(filenames) if i in correct_class_0[:, -1]
        ]

        wrong_class_1_filenames = [
            f for i, f in enumerate(filenames) if i in wrong_class_1[:, -1]
        ]
        correct_class_1_filenames = [
            f for i, f in enumerate(filenames) if i in correct_class_1[:, -1]
        ]

        class_0_keywords = extract_keywords(
            " ".join([f for i, f in enumerate(captions) if i in wrong_class_0[:, -1]])
        )
        class_1_keywords = extract_keywords(
            " ".join([f for i, f in enumerate(captions) if i in wrong_class_1[:, -1]])
        )

        with torch.no_grad():
            sim_class_0 = torch.ones(size=(20,), device=self.device)
            if len(wrong_class_0_filenames) > 0 and len(correct_class_0_filenames) > 0:
                sim_class_0 = self.forward_similarity(
                    wrong_class_0_filenames, correct_class_0_filenames, class_0_keywords
                )

            sim_class_1 = torch.ones(size=(20,), device=self.device)
            if len(wrong_class_1_filenames) > 0 and len(correct_class_1_filenames) > 0:
                sim_class_1 = self.forward_similarity(
                    wrong_class_1_filenames, correct_class_1_filenames, class_1_keywords
                )

        for sim_k, k in zip(sim_class_0, class_0_keywords):
            for i, c in enumerate(captions):
                if is_keyword_in_caption(c, k) and not is_correct[i]:
                    # print(sim_k.cpu().item(), k, i)
                    # L += [loss[i] * sim_k]
                    loss[i] *= sim_k.item()

        for sim_k, k in zip(sim_class_1, class_1_keywords):
            for i, c in enumerate(captions):
                if is_keyword_in_caption(c, k) and not is_correct[i]:
                    # print(sim_k.cpu().item(), k, i)
                    # L += [loss[i] * sim_k]
                    loss[i] *= sim_k.item()
        return loss
