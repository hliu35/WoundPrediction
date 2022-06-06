# Wound Prediction
Capstone Project


Video Generation
https://arxiv.org/pdf/2107.13766.pdf

Latent Interpolation
https://openreview.net/pdf?id=XOjv2HxIF6i

Discriminator
https://arxiv.org/pdf/2104.00567.pdf



`data_augmented` folder is for training only. 



To run the user-friendly interface
```
cd main/
python3 train.py <optional arguments here>
```





### Key Takeaways from Implementation
* Only augment the data in properties your model should be invariant to.

* Convolutions and Deconvolutions are usually better than Fully Connected layers for image generation. [Help received from](https://github.com/TeeyoHuang/conditional-GAN)

* Don't use something just because it looks fancy or new.
Understand fully before putting into action.
Use `tanh` at output layer only when image pixels are normalized to range [-1, 1].
[Help received from](https://stackoverflow.com/questions/44525338/use-of-tanh-in-the-output-layer-of-generator-network)

* The Generator or Discriminator in a GAN can get 0 loss during training. This must be avoided. [Help received from](https://www.reddit.com/r/MachineLearning/comments/5asl74/discussion_discriminator_converging_to_0_loss_in/)

* VAEs have explicit distributions, GANs have implicit distributions. [Source](https://ai.stackexchange.com/questions/8885/why-is-the-variational-auto-encoders-output-blurred-while-gans-output-is-crisp)

