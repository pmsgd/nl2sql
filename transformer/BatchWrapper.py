# -*- coding: utf-8 -*-

import transformer.Constants as Constants
import torch

class BatchWrapper:
    def __init__(self, dl, fields, device=None):
        self.dl = dl
        self.fields = fields
        self.device = device

        self.xpad = self.fields[0][1].vocab.stoi[Constants.PAD_WORD]
        self.ypad = Constants.PAD

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.fields[0][0])

            if len(self.fields) > 1:
                y = getattr(batch, self.fields[1][0])
                yield self.paired_collate_fn(x, y)
            else:
                yield self.collate_fn(x, self.xpad)

    def __len__(self):
        return len(self.dl)

    def paired_collate_fn(self, src_insts, tgt_insts):
        src_insts = self.collate_fn(src_insts, self.xpad)
        tgt_insts = self.collate_fn(tgt_insts, self.ypad)
        return (*src_insts, *tgt_insts)

    def collate_fn(self, insts, pad):
        ''' Pad the instance to the max seq length in batch '''

        batch_seq = torch.t(insts)

        batch_pos = torch.LongTensor([
            [pos_i+1 if w_i != pad else 0
             for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

        batch_seq = torch.LongTensor(batch_seq)

        return batch_seq.to(self.device), batch_pos.to(self.device)
