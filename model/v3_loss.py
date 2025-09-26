import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mpd(x):
    """
    Mean pairwise distance
    x is a tensor of shape (b, n, d) or (n, d)
    """
    if len(x.shape) == 2:
        n, d = x.shape
        x1 = x.unsqueeze(0).expand(n, n, d)
        x2 = x.unsqueeze(1).expand(n, n, d)
        stack = torch.stack([x1, x2], dim=0)
        pairwise_d = torch.norm(stack[0] - stack[1], dim=-1)
        mpd = torch.sum(torch.sum(pairwise_d, dim=0), dim=0) / (n * (n - 1))
        return mpd  # scalar
    elif len(x.shape) == 3:
        b, n, d = x.shape
        x1 = x.unsqueeze(1).expand(b, n, n, d)
        x2 = x.unsqueeze(2).expand(b, n, n, d)
        stack = torch.stack([x1, x2], dim=1)
        pairwise_d = torch.norm(stack[:, 0] - stack[:, 1], dim=-1)
        mpd = torch.sum(torch.sum(pairwise_d, dim=1), dim=1) / (n * (n - 1))
        return mpd  # (b)


class V3Loss:
    def __init__(self, config):
        """
        config: a dict of loss config. Must contain key "weights".
        """
        super(V3Loss, self).__init__()
        self.config = config
        self.eps = 1e-5

        if "supersample_content" in config or "widen_style" in config:
            self.compute_loss = self._compute_loss_adapted
        else:
            self.compute_loss = self._compute_loss_pure

    def _compute_loss_pure(
        self,
        x_hat,
        z_c,
        z_c_vq,
        commit_loss,
        z_s,
        x,
    ):
        """
        Standard form of V3 loss

        Output: a dict of losses.
        x_hat: (batch_size, n_fragments, W, H), of course it can be other shapes
        z_c: (batch_size, n_fragments, d_emb_c)
        z_s: (batch_size, n_fragments, d_emb_s)
        vq_indices: (batch_size, n_fragments)
        vq_commit_loss: already computed in VQ
        y: (batch_size, n_fragments, W, H), of course it can be other shapes
        """
        # reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # compute statistics
        content_frag_var = torch.mean(mpd(z_c) / 2 + mpd(z_c_vq) / 2, dim=0)
        content_samp_var = (
            mpd(torch.mean(z_c, dim=1)) / 2 + mpd(torch.mean(z_c_vq, dim=1)) / 2
        )
        style_frag_var = torch.mean(mpd(z_s), dim=0)
        style_samp_var = mpd(torch.mean(z_s, dim=1))

        # compute the losses using the relative variability difference
        r = self.config["relativity"]

        content_loss = F.relu(r - content_frag_var / (content_samp_var + self.eps)) / r
        style_loss = F.relu(r - style_samp_var / (style_frag_var + self.eps)) / r
        sample_loss = F.relu(r - style_samp_var / (content_samp_var + self.eps)) / r
        fragment_loss = F.relu(r - content_frag_var / (style_frag_var + self.eps)) / r

        total_loss = 0
        for k, v in self.config["weights"].items():
            if locals().get(k) is None:
                continue
            total_loss = total_loss + v * locals()[k]
        assert isinstance(total_loss, torch.Tensor)

        losses = {
            "recon_loss": recon_loss,
            "content_loss": content_loss,
            "style_loss": style_loss,
            "sample_loss": sample_loss,
            "fragment_loss": fragment_loss,
            "commit_loss": commit_loss,
            "total_loss": total_loss,
        }

        if torch.isnan(total_loss):
            print(losses)
            raise ValueError("Loss is NaN!")
        if torch.isinf(total_loss):
            print(losses)
            raise ValueError("Loss is Inf!")

        return losses

    def _compute_loss_adapted(
        self,
        x_hat,
        z_c,
        z_c_vq,
        vq_commit_loss,
        z_s,
        x,
    ):
        """
        An adapted form of V3 loss that allows supersampling content and widening style.

        Intuition behind supersampling content and widening style: We inevitably have attention residue when observing things. This is because we can't always observe everything at once. In some situations this attention residue is beneficial --- sometimes we need to check different samples together to make sense of both, or glance through multiple fragments in a sample to perceive the overall style.

        Output: a dict of losses.
        x_hat: (batch_size, n_fragments, W, H), of course it can be other shapes
        z_c: (batch_size, n_fragments, d_emb_c)
        z_s: (batch_size, n_fragments, d_emb_s)
        vq_indices: (batch_size, n_fragments)
        vq_commit_loss: already computed in VQ
        y: (batch_size, n_fragments, W, H), of course it can be other shapes
        """
        # reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # Compute statistics
        content_frag_var = torch.mean(mpd(z_c) / 2 + mpd(z_c_vq) / 2, dim=0)

        # supersample by concatenating for higher content coverage
        if "supersample_content" in self.config and self.config["supersample_content"]:
            # supersample and compute the variability
            z_c = V3Loss.supersample(z_c)
            z_c_vq = V3Loss.supersample(z_c_vq)
            content_samp_var = (
                mpd(torch.mean(z_c, dim=1)) / 2 + mpd(torch.mean(z_c_vq, dim=1)) / 2
            )
            # restore the original shape
            z_c = z_c.reshape(z_s.shape[0], z_s.shape[1], z_c.shape[-1])
            z_c_vq = z_c_vq.reshape(z_s.shape[0], z_s.shape[1], z_c_vq.shape[-1])
        else:
            content_samp_var = (
                mpd(torch.mean(z_c, dim=1)) / 2 + mpd(torch.mean(z_c_vq, dim=1)) / 2
            )

        if (
            "widen_style" in self.config and self.config["widen_style"]
        ):  # smooth by averaging for style widening
            if isinstance(self.config["widen_style"], int):
                radius = int((self.config["widen_style"] - 1) / 2)
            elif isinstance(self.config["widen_style"], bool):
                radius = 1
            z_s_padded = F.pad(z_s, (0, 0, radius, radius), mode="replicate")
            z_s_widened = z_s_padded.clone()
            for i in range(-radius, radius + 1):
                z_s_widened += z_s_padded.roll(i, dims=1)
            z_s_widened = z_s_widened / (2 * radius + 1)
            style_frag_var = torch.mean(mpd(z_s_widened), dim=0)
        else:
            style_frag_var = torch.mean(mpd(z_s), dim=0)

        if "widen_style" in self.config and self.config["widen_style"]:
            style_samp_var = mpd(torch.mean(z_s_widened, dim=1))
        if "supersample_content" in self.config and self.config["supersample_content"]:
            # concatenate z_s too, only to fairly compare with content_samp_var
            z_s = V3Loss.supersample(z_s)
            style_samp_var_ss = mpd(torch.mean(z_s, dim=1))
            z_s = z_s.reshape(z_c.shape[0], z_c.shape[1], z_s.shape[-1])
        else:
            style_samp_var = mpd(torch.mean(z_s, dim=1))

        # compute the loss using the relative variance difference
        r = self.config["relativity"]

        content_loss = F.relu(r - content_frag_var / (content_samp_var + self.eps)) / r
        style_loss = F.relu(r - style_samp_var / (style_frag_var + self.eps)) / r
        if "supersample_content" in self.config and self.config["supersample_content"]:
            sample_loss = (
                F.relu(r - style_samp_var_ss / (content_samp_var + self.eps)) / r
            )
        else:
            sample_loss = F.relu(r - style_samp_var / (content_samp_var + self.eps)) / r
        fragment_loss = F.relu(r - content_frag_var / (style_frag_var + self.eps)) / r

        total_loss = 0
        for k, v in self.config["weights"].items():
            if locals().get(k) is None:
                continue
            total_loss = total_loss + v * locals()[k]
        assert isinstance(total_loss, torch.Tensor)

        losses = {
            "recon_loss": recon_loss,
            "style_loss": style_loss,
            "content_loss": content_loss,
            "cross_batch_loss": sample_loss,
            "cross_frag_loss": fragment_loss,
            "commit_loss": vq_commit_loss,
            "total_loss": total_loss,
        }

        if torch.isnan(total_loss):
            print(losses)
            raise ValueError("Loss is NaN!")
        if torch.isinf(total_loss):
            print(losses)
            raise ValueError("Loss is Inf!")

        return losses

    @staticmethod
    def supersample(z):
        """
        Used for (content) supersampling
        If a sample contains too few fragments, we can concatenate samples to get larger samples.
            Pro: We get a higher coverage of the content vocabulary.
            Con: The styles are likely contaminated.
        Find the closest two factors of total number of fragments and use them as the new shape.
        z: [batch_size, n_fragments, d_emb]
        """
        b, n, d = z.shape
        n_fragments_all = b * n
        for i in range(int(math.sqrt(n_fragments_all)), 0, -1):
            if n_fragments_all % i == 0:
                break
        b_prime, n_prime = n_fragments_all // i, i
        z = z.reshape(b_prime, n_prime, d)

        return z
