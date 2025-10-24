---
title: "ELI5: Transformers and Decoders"
date: 2025-10-22
draft: false
description: "a description"
font_family: "Monospace"
tags: ["transformers", "huggingface", "study notes"]
---

I always find the idea of 'decoder' very confusing, specially when it gets topped by an encoder. Most of the articles on transformers focuses on attention mechanisms, using either [<ins>the OG transformer</ins>](https://peterbloem.nl/blog/transformers) or [<ins>the classic BERT</ins>](https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/) as examples. They would spend lot of time talking about embeddings & attetions on the encoding side, and skipped most the decoding by saying 'well you just do the same thing again and there you have it!'. Well that's not very helpful isn't it. Don't get me wrong, there are a lot of very good learning materials out there, for example the amazing [<ins>interactive transofmer explainer</ins>](https://poloclub.github.io/transformer-explainer). However, I always find these heavy tutorials not very suitable for my very short attention span or the autism tendency of getting lost in details.<br>
<br>
Finally, I've decided to bite the bullet and spend some time have a read through the source code of [<ins>THE encoder-decoder everyone on the steet are talking about</ins>](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py). I could not emphasis how much I appretiate HuggingFace's [<ins>maximalist coding choice</ins>](https://huggingface.co/blog/transformers-design-philosophy), where the entire model architecture is contained inside one single .py file. However, turns out it's still a rather painful process nevertheless. At least, I didn't experience the pain of come accross `import tensorflow` followed by one single if-else check [<ins>inside dataset iterator function</ins>](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/data/data_collator.py#L742)'.<br>
<br>

Here in this post, I'll very quickly go through some of the stuffs I've learned, and try to explain them as clearly as I can.<br>

## A very quick ELI5 on attention mechanism
In a very simplistic term, a trnsformer neural network can be seen as some sort of universal function approximator. That is, it's capacable of 'approximate' other formulas/ functions with certain degree of accuracy, providing the model itself is big enough ([<ins>'universal approximation theorem'</ins>](https://en.wikipedia.org/wiki/Universal_approximation_theorem)). The basic idea is that, a good model does not always need to be descriptive about underlying mechanisms, so long we are only interested in the inputs/ outputs.<br>

<br>
