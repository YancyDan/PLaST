from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import (DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX,
                          LLAST_AUDIO_PADDING_TOKEN_INDEX)


def plast_audiomask_mel_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:
    """Add audio tokens and conduct padding operation."""
    input_ids = []
    src_ids = []
    src_ids_rep = []
    audio_features = []
    speech_fearture_lens = []
    raw_audios = []
    labels = []
    feats_lens = []
    has_audio = any(inst.get('audio_tokens') is not None for inst in instances)

    if has_audio:
        audio_tokens = []
    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if has_audio:
            audio_tokens.append(example['audio_tokens'])
        feats_lens.append(torch.tensor(example['audio_lens']))
        src_ids.append(torch.tensor(example['src_ids']))
        if 'src_ids_rep' in example:
            src_ids_rep.append(torch.tensor(example['src_ids_rep']))
        if 'audio_feature' in example:
            audio_features.append(example['audio_feature'])
            speech_fearture_lens.append(torch.tensor(example['feature_len']))
        if 'raw_audio' in example:
            raw_audios.append(example['raw_audio'])
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        # padding audio tokens
        padded_audio_tokens = pad_sequence(
            audio_tokens,
            batch_first=True,
            padding_value=LLAST_AUDIO_PADDING_TOKEN_INDEX)
        src_ids = pad_sequence(
            src_ids, batch_first=True, padding_value=pad_index)
        if len(src_ids_rep) != 0:
            src_ids_rep = pad_sequence(
                src_ids_rep, batch_first=True, padding_value=pad_index)
        if len(audio_features) != 0:
            audio_features = pad_sequence(
                audio_features, batch_first=True, padding_value=pad_index)
        if len(raw_audios) != 0:
            raw_audios = pad_sequence(
                raw_audios, batch_first=True, padding_value=pad_index)

    else:
        input_ids = torch.stack(input_ids)
        src_ids = torch.stack(src_ids)
        if len(src_ids_rep) != 0:
            src_ids_rep = torch.stack(src_ids_rep)
        if len(audio_features) != 0:
            audio_features = torch.stack(audio_features)
        labels = torch.stack(labels)
        padded_audio_tokens = torch.stack(audio_tokens)

    data_dict = {
        'input_ids': input_ids,
        'src_ids': src_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels
    }

    if len(src_ids_rep) != 0:
        data_dict['src_ids_rep'] = src_ids_rep
        
    if len(audio_features) != 0:
        data_dict['audio_features'] = audio_features
        data_dict['feature_lens'] = torch.stack(speech_fearture_lens)
        
    if len(raw_audios) != 0:
        data_dict['raw_audios'] = raw_audios

    if has_audio:
        data_dict['audio_tokens'] = padded_audio_tokens
        data_dict['audio_lens'] = torch.stack(feats_lens)

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': instances}
