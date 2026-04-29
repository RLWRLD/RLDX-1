import os

import torch
import torch.nn as nn


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()

rank = local_rank


class LayerWrapper(nn.Module):
    def __init__(
        self, layer, layer_idx, internal_projection=4, img_pattern=[151652], motion_token=0
    ):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.internal_projection = internal_projection
        self.motion_token = motion_token
        self.img_pattern = img_pattern
        assert motion_token == 1

    def get_removing_indices(self, hidden_states, input_ids, num_views=None):
        pat_len = len(self.img_pattern)

        windows = input_ids.unfold(dimension=1, size=pat_len, step=1)
        pattern_tensor = torch.tensor(self.img_pattern, device=hidden_states.device).view(1, 1, -1)
        matches = (windows == pattern_tensor).all(dim=-1)

        match_lists = [
            torch.nonzero(matches[b], as_tuple=False).squeeze(-1)
            for b in range(hidden_states.shape[0])
        ]
        begin_idx = torch.tensor(
            [m[0] for m in match_lists], device=hidden_states.device
        ).unsqueeze(1)

        if num_views is not None:
            end_idx = torch.tensor(
                [m[-1 * num_view] for m, num_view in zip(match_lists, num_views)],
                device=hidden_states.device,
            ).unsqueeze(1)
        else:
            end_idx = torch.tensor(
                [m[-1] for m in match_lists], device=hidden_states.device
            ).unsqueeze(1)

        # \end_idx   = torch.tensor([m[-1 * self.num_views] for m in match_lists], device=hidden_states.device).unsqueeze(1)

        return begin_idx, end_idx

    def left_pad_emb_list(self, emb_list):
        rev = [e.flip(0) for e in emb_list]
        padded_rev = torch.nn.utils.rnn.pad_sequence(rev, batch_first=True, padding_value=0)
        return padded_rev.flip(1)

    def forward(self, hidden_states, input_ids, *args, **kwargs):
        if "image_wise_encoding" in kwargs and isinstance(
            kwargs["image_wise_encoding"], torch.Tensor
        ):
            if kwargs["image_wise_encoding"].shape[0] > 1:
                kwargs["image_wise_encoding"] = bool(kwargs["image_wise_encoding"][0])
            else:
                kwargs["image_wise_encoding"] = kwargs["image_wise_encoding"].bool().item()
        if "image_wise_encoding" in kwargs and kwargs["image_wise_encoding"]:
            num_views = kwargs["num_views"]
        else:
            num_views = None

        bsz, seq_len, dim = hidden_states.shape

        is_incremental = (
            "cache_position" in kwargs and kwargs["cache_position"] is not None and seq_len == 1
        )
        if self.layer_idx == self.internal_projection and not is_incremental:
            device = hidden_states.device

            token_indices = torch.arange(seq_len, device=device).view(1, -1).expand(bsz, -1)
            begin_idx, end_idx = self.get_removing_indices(
                hidden_states, input_ids, num_views=num_views
            )

            compress_mask = (end_idx > begin_idx).reshape(-1)

            keep_mask_front = token_indices < begin_idx
            keep_mask_back = token_indices >= end_idx

            # Drop motion module tokens (inserted before image tokens) at internal_projection
            motion_drop_info = kwargs.get("motion_drop_info", None)
            motion_drop_mask = torch.zeros_like(keep_mask_front)  # (bsz, seq_len)
            if motion_drop_info is not None and motion_drop_info["count"] > 0:
                ms = motion_drop_info["start"]
                mc = motion_drop_info["count"]
                motion_drop_mask = (token_indices >= ms) & (token_indices < ms + mc)
                keep_mask_front = keep_mask_front & ~motion_drop_mask
                kwargs["motion_drop_info"] = None  # consumed

            # Old-frame image tokens to compress (exclude motion module)
            drop_mask = ~(keep_mask_front | keep_mask_back) & ~motion_drop_mask

            motion_token = (
                (hidden_states * drop_mask.unsqueeze(-1)).sum(dim=1)
                / drop_mask.sum(dim=1, keepdim=True).clamp(min=1)
            ).reshape(bsz, self.motion_token, -1)

            hidden_states = [
                torch.cat(
                    [
                        hidden_states[b][keep_mask_front[b]],
                        motion_token[b]
                        if compress_mask[b]
                        else torch.tensor(
                            [], device=hidden_states.device, dtype=hidden_states.dtype
                        ),
                        hidden_states[b][keep_mask_back[b]],
                    ],
                    dim=0,
                )
                for b in range(bsz)
            ]

            hidden_states = self.left_pad_emb_list(hidden_states)

            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                att_list = [
                    torch.cat(
                        [
                            kwargs["attention_mask"][b][keep_mask_front[b]],
                            torch.ones(
                                1,
                                device=kwargs["attention_mask"].device,
                                dtype=kwargs["attention_mask"].dtype,
                            )
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["attention_mask"].device,
                                dtype=kwargs["attention_mask"].dtype,
                            ),
                            kwargs["attention_mask"][b][keep_mask_back[b]],
                        ]
                    )
                    for b in range(bsz)
                ]
                kwargs["attention_mask"] = self.left_pad_emb_list(att_list)
            if "attention_mask_2" in kwargs and kwargs["attention_mask_2"] is not None:
                att_list_2 = [
                    torch.cat(
                        [
                            kwargs["attention_mask_2"][b][keep_mask_front[b]],
                            torch.ones(
                                1,
                                device=kwargs["attention_mask_2"].device,
                                dtype=kwargs["attention_mask_2"].dtype,
                            )
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["attention_mask_2"].device,
                                dtype=kwargs["attention_mask_2"].dtype,
                            ),
                            kwargs["attention_mask_2"][b][keep_mask_back[b]],
                        ]
                    )
                    for b in range(bsz)
                ]
                kwargs["attention_mask_2"] = self.left_pad_emb_list(att_list_2)

            if "position_ids" in kwargs.keys() and kwargs["position_ids"] is not None:
                position_ids = kwargs["position_ids"]
                if position_ids.dim() == 2:
                    pos_list = [
                        torch.cat(
                            [
                                position_ids[b][keep_mask_front[b]],
                                position_ids[b][begin_idx[b] : begin_idx[b] + 1]
                                if compress_mask[b]
                                else position_ids[b][:0],
                                position_ids[b][keep_mask_back[b]],
                            ]
                        )
                        for b in range(bsz)
                    ]
                    kwargs["position_ids"] = self.left_pad_emb_list(pos_list)
                elif position_ids.dim() == 3:
                    # Keep the leading rotary dimension (e.g. 3) and compress on seq axis.
                    pos_list = [
                        torch.cat(
                            [
                                position_ids[:, b, keep_mask_front[b]],
                                position_ids[:, b, begin_idx[b] : begin_idx[b] + 1]
                                if compress_mask[b]
                                else position_ids[:, b, :0],
                                position_ids[:, b, keep_mask_back[b]],
                            ],
                            dim=-1,
                        )
                        for b in range(bsz)
                    ]
                    pos_list_for_pad = [p.transpose(0, 1) for p in pos_list]  # [seq, rope_dim]
                    padded = self.left_pad_emb_list(pos_list_for_pad)  # [bs, max_seq, rope_dim]
                    kwargs["position_ids"] = padded.permute(2, 0, 1)  # [rope_dim, bs, max_seq]
                else:
                    raise ValueError(f"Unsupported position_ids shape: {position_ids.shape}")

            if "position_embeddings" in kwargs.keys() and kwargs["position_embeddings"] is not None:
                emb_x_list = [
                    torch.cat(
                        [
                            kwargs["position_embeddings"][0][b][keep_mask_front[b]],
                            kwargs["position_embeddings"][0][b][begin_idx[b] : begin_idx[b] + 1]
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["position_embeddings"][0].device,
                                dtype=kwargs["position_embeddings"][0].dtype,
                            ),
                            kwargs["position_embeddings"][0][b][keep_mask_back[b]],
                        ],
                        dim=0,
                    )
                    for b in range(bsz)
                ]

                emb_y_list = [
                    torch.cat(
                        [
                            kwargs["position_embeddings"][1][b][keep_mask_front[b]],
                            kwargs["position_embeddings"][1][b][begin_idx[b] : begin_idx[b] + 1]
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["position_embeddings"][0].device,
                                dtype=kwargs["position_embeddings"][0].dtype,
                            ),
                            kwargs["position_embeddings"][1][b][keep_mask_back[b]],
                        ],
                        dim=0,
                    )
                    for b in range(bsz)
                ]

                emb_x_padded = self.left_pad_emb_list(emb_x_list)
                emb_y_padded = self.left_pad_emb_list(emb_y_list)
                kwargs["position_embeddings"] = (emb_x_padded, emb_y_padded)

            if "cache_position" in kwargs and kwargs["cache_position"] is not None:
                kwargs["cache_position"] = kwargs["cache_position"][: hidden_states.shape[1]]

        return self.layer(hidden_states, *args, **kwargs), kwargs
