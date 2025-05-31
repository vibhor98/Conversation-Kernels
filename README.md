# Conversation Kernels: A Flexible Mechanism to Learn Relevant Context for Online Conversation Understanding

**Vibhor Agarwal**, Arjoo Gupta, Suparna De, and Nishanth Sastry, "Conversation Kernels: A Flexible Mechanism to Learn Relevant Context for Online Conversation Understanding", International AAAI Conference on Web and Social Media (ICWSM), 2025.

## Abstract
Understanding online conversations has attracted research attention with the growth of social networks and online discussion forums. Content analysis of posts and replies in online conversations is difficult because each individual utterance is usually short and may implicitly refer to other posts within the same conversation. Thus, understanding individual posts requires capturing the conversational context and dependencies between different parts of a conversation tree and then encoding the context dependencies between posts and comments/replies into the language model.

To this end, we propose a general-purpose mechanism to discover appropriate conversational context for various aspects about an online post in a conversation, such as whether it is informative, insightful, interesting or funny. Specifically, we design two families of Conversation Kernels, which explore different parts of the neighborhood of a post in the tree representing the conversation and through this, build relevant conversational context that is appropriate for each task being considered. We apply our developed method to conversations crawled from slashdot.org, which allows users to apply highly different labels to posts, such as ‘insightful’, ‘funny’, etc., and therefore provides an ideal experimental platform to study whether a framework such as Conversation Kernels is general-purpose and flexible enough to be adapted to disparately different conversation understanding tasks.

We perform extensive experiments and find that context-augmented conversation kernels can significantly outperform transformer-based baselines, with absolute improvements in accuracy up to 20% and up to 19% for macro-F1 score. Our evaluations also show that conversation kernels outperform state-of-the-art large language models including GPT-4. We also showcase the generalizability and demonstrate that conversation kernels can be a general-purpose approach that flexibly handles distinctly different conversation understanding tasks in a unified manner.

The paper PDF is available [here](https://arxiv.org/abs/2505.20482)!

## Overview
**GraphNLI** is a graph-based deep learning architecture for polarity prediction, which captures both the local and the global context of the online debates through graph walks.

<div align="center">
  <img src="https://github.com/vibhor98/Conversation-Kernels/tree/main/figures/Conversation_Kernels.pdf">
</div>

## Slashdot Dataset
To get the Slashdot dataset of online conversations, please request the dataset [here](https://netsys.surrey.ac.uk/datasets/slashdot/) by filling the form.

## Citation
If you find this paper useful in your research, please consider citing:
```
@article{agarwal2025conversation,
    title={Conversation Kernels: A Flexible Mechanism to Learn Relevant Context for Online Conversation Understanding},
    author={Agarwal, Vibhor and Gupta, Arjoo and De, Suparna and Sastry, Nishanth},
    journal={arXiv preprint arXiv:2505.20482},
    year={2025}
}
```
