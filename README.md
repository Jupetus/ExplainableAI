# ExplainableAI


Collection of experiments that can be used for explainable AI

## Implementations
---

### Input times the gradient

| Original image | Input x Gradient |
| ------------- | ------------- |
| TODO | TODO |

---

### Adversial attack

As implemented in https://arxiv.org/abs/1412.6572

Adversial attack aims to introduce a noice which will cause our prediction to go wrong.

| Original Image | Adversial Attack |
| ------------- | ------------- |
| ![Original](Results/adversial_original.png?raw=true) | ![Attack](Results/adversial_attack.png?raw=true) |

---

### ScoreCam

As implemented in https://arxiv.org/pdf/1910.01279.pdf

Scorecam aims to make a hypothesis what region of the image our network will make its decision based on. Whiter the pixels are in the picture, more attention the model pays to the region. 

| Original Image | Scorecam Focus |
| ------------- | ------------- |
| ![Original](Results/scorecam_in.png?raw=true)  | ![Attack](Results/scorecam_out.png?raw=true)  |

