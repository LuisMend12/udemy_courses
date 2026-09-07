# Neural Machine Translation (English -> Spanish)

A sequence-to-sequence Transformer built from scratch in PyTorch (custom multi-head attention,
positional encoding, and encoder/decoder stacks - no `nn.Transformer` shortcut), trained to
translate short English sentences into Spanish.

- **Notebook**: [`en_es_transformer.ipynb`](en_es_transformer.ipynb)
- **Data**: [`opus_books`](https://huggingface.co/datasets/opus_books) English-Spanish parallel
  corpus, loaded via Hugging Face `datasets`.
- **Tokenization**: word-level (regex tokenizer), per-language vocab built from the training
  split.

## Running it

Open the notebook and run all cells top to bottom. It installs `datasets` and `sacrebleu` itself.
On CPU, training the default config (~29k sentence pairs, 12 epochs) takes roughly 1-1.5 hours;
a GPU brings that down to a few minutes. To iterate faster, shrink the dataset subset
(`pairs = pairs[:30000]`) or `N_EPOCHS` in the training section.

The notebook covers: data loading/cleaning, vocab building, the Transformer implementation,
training loop, greedy-decoding inference, and BLEU evaluation - plus notes on extending it
(subword tokenization, beam search, or swapping in a pretrained model like
`Helsinki-NLP/opus-mt-en-es` for higher translation quality).

## Results (already baked into the notebook)

12 epochs on ~29k sentence pairs (CPU, ~87 min): train loss 5.85 -> 3.96, val loss 5.16 -> 4.39
(still slowly improving at epoch 12 - more epochs/data would help further). Corpus BLEU on 300
held-out pairs: **7.57**. Sample translations:

| English | Model output |
|---|---|
| hello, how are you? | ¿ cómo estáis ? |
| i love you | te amo |
| the house is very big | es la casa de la casa . |
| where is the book? | ¿ dónde está el libro ? |
| good morning | mañana |

Short, common phrases translate reasonably; longer/compositional sentences ("the house is very
big") still go off the rails - expected for a small word-level model trained this briefly.
