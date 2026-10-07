"""
Downloads a set of Wikipedia articles on machine learning topics
into a local folder. Run this once before build_index.py.
"""

import os
import wikipediaapi

SUBJECTS = [
    # fundamentals
    "Machine learning",
    "Artificial intelligence",
    "Deep learning",
    "Supervised learning",
    "Unsupervised learning",
    "Overfitting",
    "Regularization (mathematics)",
    "Cross-validation (statistics)",
    "Bias-variance tradeoff",
    "Feature engineering",
    "Curse of dimensionality",

    # neural networks
    "Neural network (machine learning)",
    "Backpropagation",
    "Gradient descent",
    "Stochastic gradient descent",
    "Activation function",
    "Rectifier (neural networks)",
    "Batch normalization",
    "Dilution (neural networks)",
    "Vanishing gradient problem",
    "Multilayer perceptron",

    # architectures
    "Convolutional neural network",
    "Recurrent neural network",
    "Long short-term memory",
    "Residual neural network",
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Autoencoder",
    "Generative adversarial network",
    "U-Net",

    # computer vision
    "Computer vision",
    "Object detection",
    "Image segmentation",
    "Optical character recognition",
    "Data augmentation",
    "Transfer learning",

    # NLP
    "Natural language processing",
    "Word embedding",
    "Word2vec",
    "BERT (language model)",
    "Large language model",
    "Tokenization (lexical analysis)",

    # classical ML
    "Random forest",
    "Decision tree learning",
    "Gradient boosting",
    "Support vector machine",
    "K-means clustering",
    "K-nearest neighbors algorithm",
    "Principal component analysis",
    "Logistic regression",
    "Linear regression",
    "Naive Bayes classifier",

    # evaluation
    "Precision and recall",
    "Receiver operating characteristic",
    "Confusion matrix",
    "F-score",
    "Mean squared error",

    # other
    "Reinforcement learning",
    "Q-learning",
    "Hyperparameter optimization",
    "Ensemble learning",
    "Semi-supervised learning",
]

DOCS_DIR = "docs"
USER_AGENT = "ml-assistant-rag/1.0"


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")

    saved = 0
    for subject in SUBJECTS:
        page = wiki.page(subject)
        if not page.exists():
            print("missing:", subject)
            continue

        path = os.path.join(DOCS_DIR, subject.replace(" ", "_") + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(page.text)
        saved += 1

    print(f"saved {saved} of {len(SUBJECTS)} articles to {DOCS_DIR}/")


if __name__ == "__main__":
    main()
