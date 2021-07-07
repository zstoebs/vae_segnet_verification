# Verification of a VAE and SegNet using NNV

## Abstract
This report details a study on deep generator network verification using the Neural Network Verification (NNV) toolbox – a set-based deep neural network (DNN) and learning-enabled verification framework. NNV defines exact and over-approximate reachability algorithms for several input set representations, such as zonotopes, star sets, and abstract-domain polytopes. In this project, NNV’s reachability algorithms were applied to an MNIST convolutional classifier network that takes as input the output of a variational autoencoder (VAE) and a SegNet to verify the classifier’s robustness in correctly classifying a fake image. The input set representations used were the ImageZonotope and the ImageStar. The output set methods were the approximated zonotope, star, and abstract polytope. When the output of the generator network was left untainted, the classifier robustly classified the fake image in most cases. Similarly, attacking the brightness of the real and fake images expectedly confused the model more than the unperturbed images. Large attacks resulted in greater confusion, while the classifier was still robust in a few cases for small brightness attacks. Raw input classification was not more robust than the fake inputs, which are clearly imperfect, while brightness attacks can confound the model.

## Content
- train_segnet.m: trains a SegNet on MNIST
- train_VAE.m: trains a VAE on MNIST
- verify_segnet.m: attempts verification of SegNet given perturbed or unperturbed input
- verify_VAE.m: attempts verification of VAE given perturbed or unperturbed input

Other files are ported or modified from [MATLAB's VAE demo](https://www.mathworks.com/help/deeplearning/ug/train-a-variational-autoencoder-vae-to-generate-images.html). 

## References 

X. Huang et al., “A Survey of Safety and Trustworthiness of Deep Neural Networks: Verification, Testing, Adversarial Attack and Defence, and Interpretability ∗,” arXiv, pp. 0–94, 2018.

C. Liu, T. Arnon, C. Lazarus, C. Barrett, and M. J. Kochenderfer, “Algorithms for verifying deep neural networks,” arXiv, pp. 1–126, 2019, doi: 10.1561/2400000035.

K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 3rd Int. Conf. Learn. Represent. ICLR 2015 - Conf. Track Proc., pp. 1–14, 2015.

H. D. Tran et al., “NNV: The Neural Network Verification Tool for Deep Neural Networks and Learning-Enabled Cyber-Physical Systems,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 12224 LNCS, pp. 3–17, 2020, doi: 10.1007/978-3- 030-53288-8_1.

Y. Lecun, L. Bottou, Y. Bengio, and P. Ha, “LeNet,” Proc. IEEE, no. November, pp. 1–46, 1998.

D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” 2nd Int. Conf. Learn. Represent. ICLR 2014 - Conf. Track Proc., no. Ml, pp. 1–14, 2014.

V. Badrinarayanan, A. Kendall, and R. Cipolla, “SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 12, pp. 2481–2495, 2017, doi: 10.1109/TPAMI.2016.2644615.

A. A. Alemi, B. Poole, I. Fische, J. V. Dillon, R. A. Saurous, and K. Murphy, “Fixing a broken elbo,” 35th Int. Conf. Mach. Learn. ICML 2018, vol. 1, pp. 245–265, 2018.

H. D. Tran et al., “Star-based reachability analysis of deep neural networks,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 11800 LNCS, pp. 670–686, 2019, doi: 10.1007/978-3-030-30942-8_39. 
