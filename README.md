# Udemy Courses

Personal notes, notebooks, and exercises from Udemy courses on machine learning, deep learning, and reinforcement learning — plus related coursework and side projects that grew out of them (university AI coursework, a quantum-kernel research paper, and a couple of simulation side projects).

## Structure

| Directory | Content |
|---|---|
| [`ML A-Z/`](ML%20A-Z) | "Machine Learning A-Z" course — data preprocessing, regression (simple/multiple linear, polynomial, SVR, decision tree, random forest), classification (KNN, SVM, kernel SVM, Naive Bayes, decision trees, random forest, logistic regression), and clustering (k-means), organized by course part with notebooks, datasets, and notes (`code_in_r/`, `notes/`) |
| [`Deep_learning_cnns/`](Deep_learning_cnns) | Deep learning course work, including a full clone of Lazy Programmer's [`machine_learning_examples`](Deep_learning_cnns/machine_learning_examples) repo (ANNs, CNNs, RNNs, NLP, recommenders, reinforcement learning, TensorFlow/PyTorch examples, etc.) |
| [`hyperparameter_optimization_for_ml/`](hyperparameter_optimization_for_ml) | Hyperparameter optimization course — a clone of the [`hyperparameter-optimization`](hyperparameter_optimization_for_ml/hyperparameter-optimization) course repo (grid/random search, Bayesian optimization, Scikit-Optimize, Hyperopt, Optuna), a Kaggle digit-recognizer dataset, personal notes, and a `gaussian_basics/` side project including a from-scratch Cholesky decomposition simulator (C++ backend/frontend + Python) |
| [`rl_course/`](rl_course) | Reinforcement learning fundamentals — Markov decision processes, dynamic programming (policy/value iteration), on-policy and off-policy Monte Carlo control, with notebooks, LaTeX notes, and a `fun_projects/` folder (e.g. an Arsenal vs. PSG RL demo with 2D/3D frontends) |
| [`ultimate_rag_bootcamp/`](ultimate_rag_bootcamp) | RAG (Retrieval-Augmented Generation) bootcamp notes, starting with an intro-to-RAG writeup |
| [`ECE469_Artificial_Intelligence/`](ECE469_Artificial_Intelligence) | Coursework for ECE-469 Artificial Intelligence at Cooper Union — homeworks (`hw1`–`hw3`), lecture slides, and two programming projects (a game-playing AI checker, an artificial neural network) |
| [`quantum_kernel_classification/`](quantum_kernel_classification) | Independent research project/paper: "When Do Quantum Kernels Help?" — a from-scratch NumPy simulation of the Havlíček et al. ZZ feature-map quantum kernel, benchmarked against classical kernels, with a NeurIPS-format writeup in `paper/` |
| [`brownian_motion_sim/`](brownian_motion_sim) | C++ Brownian motion (Wiener process) simulator producing particle trajectory data |
| [`assets/`](assets) | Standalone images (e.g. STEM-GNN architecture diagrams) referenced from notes |

## Notes

- Some subfolders (`Deep_learning_cnns/machine_learning_examples`, `hyperparameter_optimization_for_ml/hyperparameter-optimization`) are git submodules pointing at external course repositories — see [`.gitmodules`](.gitmodules).
- Course notes are largely written in LaTeX (`.tex`, compiled to `.pdf`) alongside Jupyter notebooks (`.ipynb`) and R scripts.
- `quantum_kernel_classification/` and `ECE469_Artificial_Intelligence/` aren't Udemy material — they're adjacent coursework/research kept here alongside it.
- This is a study/reference repo, not a packaged library — there's no single install step or entry point; open the relevant notebook, notes file, or project `README.md` for a given topic.
