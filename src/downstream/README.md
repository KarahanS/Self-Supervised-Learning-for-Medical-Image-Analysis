There are different [methods](https://www.youtube.com/watch?v=TDgJz0yMtFQ) to evaluate the learned representations:
1. Linear evaluation: A linear classifier (logistic regression) is trained on top of the frozen base network, and test accuracy is used as a proxy for representation quality. 
2. Non-linear evaluation: A non-linear classifier (MLP with ReLU) is trained on top of the frozen base
network, and test accuracy is used as a proxy for representation quality.
3. Semi-supervised learning (Data Efficiency): Fine-tune the whole base network on the either 1% or 10% of the labeled data without regularization. The idea here is to simulate a data efficient solution combining self-supervised and supervised approaches. After fine-tuning, we can use linear or non-linear evaluation.
4. Transfer learning: Evaluate the encoder on a different dataset than it used for self-supervised pretraining. Transfer learning is a more generic approach and it can be performed using linear/non-linear evaluation or semi-supervised learning approaches.

Success measures on downstream tasks can include (but not limited to) top-1 accuracy and top-5 accuracy.
