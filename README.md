# Text-Generation-Papers
Paper collection of __Neural Text Generation__ tasks, may include __Image Caption__ / __Summarization__

(working on it ......)

## __Survey__

- 2016 ___Tutorial on variational autoencoders___ [[pdf](https://arxiv.org/pdf/1606.05908.pdf)]

- 2017 ___Neural text generation: A practical guide___ [[pdf](https://arxiv.org/pdf/1711.09534.pdf)]

- 2018 ___Survey of the state of the art in natural language generation: Core tasks, applications and evaluation___ [[pdf](https://www.jair.org/index.php/jair/article/download/11173/26378)]

- 2018 ___Neural Text Generation: Past, Present and Beyond___ [[pdf](https://arxiv.org/pdf/1803.07133.pdf)]

- 2018 ___Survey of the state of the art in natural language generation: Core tasks, applications and evaluation___ [[link](https://www.jair.org/index.php/jair/article/view/11173)]

---

# Text Generation Model

## __General__

#### __Classic__
- 1997 ___Long short-term memory___ [[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)] __(LSTM)__

- 2003 ___A neural probabilistic language model___ [[pdf](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)] __(NNLM)__

- 2010 ___Recurrent neural network based language model___ [[pdf](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)] __(RNNLM)__

- 2014 ___Sequence to sequence learning with neural networks___ [[pdf](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)] __(seq2seq)__

- 2014 ___Neural machine translation by jointly learning to align and translate___ [[pdf](https://arxiv.org/pdf/1409.0473)] __(Attn)__

- 2014 ___Learning phrase representations using RNN encoder-decoder for statistical machine translation___ [[pdf](https://arxiv.org/pdf/1406.1078.pdf)] __(GRU)__

- 2015 ___Scheduled sampling for sequence prediction with recurrent neural networks___ [[pdf](https://papers.nips.cc/paper/5956-scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks.pdf)]



- 2017 ___Attention is all you need___ [[pdf](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)]

#### __Others__

- 2016 ___Controlling Output Length in Neural Encoder-Decoders___ [[pdf](https://arxiv.org/pdf/1609.09552.pdf?__hstc=36392319.43051b9659a07455a3db8391a8f20ea4.1480118400085.1480118400086.1480118400087.1&__hssc=36392319.1.1480118400088&__hsfp=528229161)]

- 2019 ___Non-Monotonic Sequential Text Generation___ [[pdf](https://arxiv.org/pdf/1902.02192)]



## __VAE__
- 2013 ___Auto-encoding variational bayes___ [[pdf](https://arxiv.org/pdf/1312.6114.pdf)] __(VAE)__

- 2015 ___Generating Sentences from a Continuous Space___ [[pdf](https://arxiv.org/pdf/1511.06349.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)]

- 2017 ___Toward controlled generation of text___ [[pdf](https://arxiv.org/pdf/1703.00955.pdf)]

- 2017 ___Improved variational autoencoders for text modeling using dilated convolutions___ [[pdf](https://arxiv.org/pdf/1702.08139.pdf)]

- 2017 ___Variational Attention for Sequence-to-Sequence Models___ [[pdf](https://arxiv.org/pdf/1712.08207.pdf)]

- 2018 ___Semi-Amortized Variational Autoencoders___ [[pdf](https://arxiv.org/pdf/1802.02550.pdf)]

- 2018 ___Unsupervised natural language generation with denoising autoencoders___ [[pdf](https://arxiv.org/pdf/1804.07899)]

- 2018 ___Latent alignment and variational attention___ [[pdf](https://arxiv.org/pdf/1807.03756.pdf)] __(Attn-VAE)__



## __GAN__
- 2016 ___Gans for sequences of discrete elements with the gumbel-softmax distribution___ [[pdf](https://arxiv.org/pdf/1611.04051.pdf)] __(Gumbel-Softmax)__

- 2017 ___Seqgan: Sequence generative adversarial nets with policy gradient___ [[link](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14344)] __(SeqGAN)__

- 2017 ___Maximum-likelihood augmented discrete generative adversarial networks___ [[pdf](https://arxiv.org/pdf/1702.07983)] __(MaliGAN)__

- 2017 ___Adversarial ranking for language generation___ [[pdf](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation.pdf)] __(RankGAN)__

- 2017 ___Adversarial feature matching for text generation___ [[pdf](https://arxiv.org/pdf/1706.03850.pdf)] __(TextGAN)__

- 2018 ___Maskgan: Better text generation via filling in the \____ [[pdf](https://arxiv.org/pdf/1801.07736.pdf%3C/p%3E)] __(MaskGAN)__

- 2018 ___Long text generation via adversarial training with leaked information___ [[pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16360/16061)] __(LeakGAN)__

- 2018 ___Diversity-Promoting GAN: A Cross-Entropy Based Generative Adversarial Network for Diversified Text Generation___ [[pdf](http://www.aclweb.org/anthology/D18-1428)] __(DpGAN)__

- 2018 ___SentiGAN: Generating Sentimental Texts via Mixture Adversarial Networks___ [[pdf](https://www.ijcai.org/proceedings/2018/0618.pdf)] __(SentiGAN)__


---

# Image Caption

#### __Classic__

- 2015 ___Show, attend and tell: Neural image caption generation with visual attention___ [[pdf](http://proceedings.mlr.press/v37/xuc15.pdf)] __(ATTN)__

- 2015 ___Show and tell: A neural image caption generator___ [[pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)] __(NIC)__

- 2015 ___Deep visual-semantic alignments for generating image descriptions___ [[pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)]

- 2017 ___Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning___ [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Knowing_When_to_CVPR_2017_paper.pdf)] __(Ada-ATTN)__

- 2017 ___Self-critical sequence training for image captioning___ [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf)] __(SCST)__

- 2017 ___Towards Diverse and Natural Image Descriptions via a Conditional GAN___ [[pdf](https://arxiv.org/pdf/1703.06029.pdf)]



#### __Others__

- 2018 ___Entity-aware Image Caption Generation___ [[pdf](https://arxiv.org/pdf/1804.07889.pdfs)]


# Summarization

## Reinforcement Learning

- 2018 ___A Deep Reinforced Model for Abstractive Summarization___ [[pdf](https://openreview.net/pdf?id=HkAClQgA-)]

- 2018 ___Improving Abstraction in Text Summarization___ [[pdf](https://arxiv.org/pdf/1808.07913.pdf)]

- 2018 ___Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting___ [[pdf](https://arxiv.org/pdf/1805.11080.pdf)]

- 2018 ___Multi-Reward Reinforced Summarization with Saliency and Entailment___ [[pdf](https://arxiv.org/pdf/1804.06451.pdf)]

- 2018 ___Closed-Book Training to Improve Summarization Encoder Memory___ [[pdf](https://arxiv.org/pdf/1809.04585.pdf)]