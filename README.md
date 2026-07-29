# Udemy Courses

Personal notes, notebooks, and exercises from Udemy courses on machine learning, deep learning, reinforcement learning, and related topics.

## Structure

| Directory | Content |
|---|---|
| [`ML A-Z/`](ML%20A-Z) | "Machine Learning A-Z" course — data preprocessing, regression (simple/multiple linear, polynomial, SVR, decision tree, random forest), classification (KNN, SVM, kernel SVM, Naive Bayes, decision trees, random forest, logistic regression), and clustering (k-means), organized by course part with notebooks, datasets, and notes (`code_in_r/`, `notes/`) |
| [`Deep_learning_cnns/`](Deep_learning_cnns) | Deep learning course work, including a full clone of Lazy Programmer's [`machine_learning_examples`](Deep_learning_cnns/machine_learning_examples) repo (ANNs, CNNs, RNNs, NLP, recommenders, reinforcement learning, TensorFlow/PyTorch examples, etc.) |
| [`hyperparameter_optimization_for_ml/`](hyperparameter_optimization_for_ml) | Hyperparameter optimization course — a clone of the [`hyperparameter-optimization`](hyperparameter_optimization_for_ml/hyperparameter-optimization) course repo (grid/random search, Bayesian optimization, Scikit-Optimize, Hyperopt, Optuna) plus a Kaggle digit-recognizer dataset and personal notes |
| [`rl_course/`](rl_course) | Reinforcement learning fundamentals — Markov decision processes and dynamic programming, with notebooks and LaTeX notes |
| [`ultimate_rag_bootcamp/`](ultimate_rag_bootcamp) | RAG (Retrieval-Augmented Generation) bootcamp notes, starting with an intro-to-RAG writeup |
| [`assets/`](assets) | Standalone images (e.g. STEM-GNN architecture diagrams) referenced from notes |

## Notes

- Some subfolders (`Deep_learning_cnns/machine_learning_examples`, `hyperparameter_optimization_for_ml/hyperparameter-optimization`) are full copies of external course repositories, not submodules — they carry their own `.git`/history and `README.md`.
- Course notes are largely written in LaTeX (`.tex`, compiled to `.pdf`) alongside Jupyter notebooks (`.ipynb`) and R scripts.
- This is a study/reference repo, not a packaged library — there's no single install step or entry point; open the relevant notebook or notes file for a given topic.
