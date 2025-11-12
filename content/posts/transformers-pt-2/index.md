---
title: "ELI5 Transformers (Part 1) - Generation"
date: 2025-11-11
publishDate: 2025-11-11
draft: false
summary: "(it's glorified sequence classification)"
font_family: "Monospace"
tags: ["huggingface", "AI", "ELI5", "pytorch"]
category: ["ELI5-Transformers",]
---

{{< katex >}}
![always has been](./featured.jpg "always has been")

I'm pretty sure you've already heard about someting like this before: 'generative LLM is just slightly advanced auto complete'. Or, something like 'all it does is predicting what's the most likely next word with all the previous words given'.

Today I would like to invite you think of text generation in a very different perspective, at least it's the persceptive I found helped me the most: ***text generation is glorified sequence classification.*** I'm going to use **GPT-2** model as an example, walk you through the mechanism of text generation with the source code, and show you how text generation *actually* works.


## Prep
Open termianl and type-in these codes to install depencies in case you haven't done so:
```sh
pip install transformers
```
Then in terminal, type 'python' to start python interactive REPL:
```sh
python
```
You should see something like this in your terminal:

```sh
Python 3.13.7 | packaged by conda-forge | (main, Sep  3 2025, 14:24:46) [Clang 19.1.7 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Import dependencies:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
```

*As of November 2025, Pytorch/ Apple still haven't fixed the [memory leak](https://github.com/pytorch/pytorch/issues/91368) issue on Apple Silicon devices (i.e., post-2020 Macbooks). As a result, running models with pytorch for some period of time will gets slower and slower over time. Just for demonstration purpose, I'd recomment manually set pytorch device as 'cpu' because of this. Skip this step if you were confident this won't happen. <br>Since we are just doing demonstrations, we can simly set torch device as 'cpu':<br>*

```python
torch.set_default_device('cpu')
```

And download the model:
```python
checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=128)
```



## Quick recap on transformer architecture
### 1 - Inputs <-> outputs
### 2 - 'Task Head'

## Model generation vs text classification
(Here is where I write about LM head)

## Popular text generation strategies

## 预告？？？parallel training