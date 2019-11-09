''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.BatchWrapper import BatchWrapper
from transformer.Translator import Translator
from torchtext import data as textdata
from transformer.Models import Transformer
import torch.nn as nn
import io


def load_model(opt):
    # TODO not working with save mode 'all'
    checkpoint = torch.load(opt.model + '.chkpt', map_location=opt.device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.tgt_vocab_size,
        model_opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_tgt_weight_sharing=False,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')

    return model, model_opt


def load_vocabs(opt):
    en_vocab = torch.load(opt.model + '_en.vocab', map_location=opt.device)
    sql_vocab = torch.load(opt.model + '_sql.vocab', map_location=opt.device)

    return en_vocab, sql_vocab


def load_data(opt, en_vocab):
    en = textdata.Field(tokenize='spacy', tokenizer_language='en',
                        init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD,
                        pad_token=Constants.PAD_WORD, unk_token=Constants.UNK_WORD)
    en.vocab = en_vocab
    fields = [("en", en)]

    examples = []
    with io.open(opt.src, mode='r', encoding='utf-8') as src_file:
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(textdata.Example.fromlist([src_line], fields))

    ds = textdata.Dataset(examples, fields)
    ds_iter = textdata.BucketIterator(dataset=ds,
                                      batch_size=opt.batch_size,
                                      sort_key=lambda x: len(x.en),
                                      device=opt.device)

    return BatchWrapper(ds_iter, fields, device=opt.device), en


def create_reversible_field(sql_vocab):
    sql_tokenizer = lambda x: x.split(Constants.SQL_SEPARATOR)
    sql = textdata.ReversibleField(tokenize=sql_tokenizer,
                                   init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD,
                                   pad_token=Constants.PAD_WORD, unk_token=Constants.UNK_WORD)
    sql.vocab = sql_vocab

    return sql


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('--model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('--src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('--output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('--n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # load model
    model, model_opt = load_model(opt)
    en_vocab, sql_vocab = load_vocabs(opt)

    # load data
    loader, en_field = load_data(opt, en_vocab)
    sql_field = create_reversible_field(sql_vocab)
    bos_token = sql_field.vocab[Constants.BOS_WORD]
    eos_token = sql_field.vocab[Constants.EOS_WORD]
    pad_token = sql_field.vocab[Constants.PAD_WORD]

    print('[Info] Inference start.')
    translator = Translator(opt, model, model_opt)
    with open(opt.output, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(loader, mininterval=2, desc='  - (Test)', leave=False):
                all_hyp, all_scores = translator.translate_batch(*batch,
                                                                 bos_token=bos_token,
                                                                 eos_token=eos_token,
                                                                 pad_token=pad_token)
                for idx_seqs in all_hyp:
                    pred_line = ' '.join(sql_field.reverse(torch.LongTensor(idx_seqs)))
                    f.write(pred_line + '\n')
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
