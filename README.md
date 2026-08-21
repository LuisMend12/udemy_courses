# Udemy Courses

Personal notes, notebooks, and exercises from Udemy courses on machine learning, deep learning, and reinforcement learning — plus related coursework and side projects that grew out of them (university AI coursework, a quantum-kernel research paper, and a couple of simulation side projects).

## Structure

| Directory | Content | Course |
|---|---|---|
| [`ML A-Z/`](ML%20A-Z) | "Machine Learning A-Z" course — data preprocessing, regression (simple/multiple linear, polynomial, SVR, decision tree, random forest), classification (KNN, SVM, kernel SVM, Naive Bayes, decision trees, random forest, logistic regression), clustering (k-means), and reinforcement learning basics (Upper Confidence Bound, Thompson Sampling, in `part-6-intro-to-rl/`), organized by course part with notebooks, datasets, and notes (`code_in_r/`, `notes/`) | [Machine Learning A-Z (Udemy)](https://www.udemy.com/course/machinelearning/) |
| [`Deep_learning_cnns/`](Deep_learning_cnns) | Deep learning course work, including a full clone of Lazy Programmer's [`machine_learning_examples`](Deep_learning_cnns/machine_learning_examples) repo (ANNs, CNNs, RNNs, NLP, recommenders, reinforcement learning, TensorFlow/PyTorch examples, etc.) | [Lazy Programmer's Deep Learning series](https://lazyprogrammer.me/deep-learning-courses/) |
| [`hyperparameter_optimization_for_ml/`](hyperparameter_optimization_for_ml) | Hyperparameter optimization course — a clone of the [`hyperparameter-optimization`](hyperparameter_optimization_for_ml/hyperparameter-optimization) course repo (grid/random search, Bayesian optimization, Scikit-Optimize, Hyperopt, Optuna), a Kaggle digit-recognizer dataset, personal notes, and a `gaussian_basics/` side project including a from-scratch Cholesky decomposition simulator (C++ backend/frontend + Python) | [Hyperparameter Optimization for Machine Learning (Udemy)](https://www.udemy.com/course/hyperparameter-optimization-for-machine-learning/) |
| [`rl_course/`](rl_course) | Reinforcement learning fundamentals — Markov decision processes, dynamic programming (policy/value iteration), on-policy and off-policy Monte Carlo control, with notebooks, LaTeX notes, and a `fun_projects/` folder (e.g. an Arsenal vs. PSG RL demo with 2D/3D frontends) | [Artificial Intelligence: Reinforcement Learning in Python (Udemy)](https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/) |
| [`ultimate_rag_bootcamp/`](ultimate_rag_bootcamp) | RAG (Retrieval-Augmented Generation) bootcamp notes, starting with an intro-to-RAG writeup | [Ultimate RAG Bootcamp Using LangChain, LangGraph & LangSmith (Udemy)](https://www.udemy.com/course/ultimate-rag-bootcamp-using-langchainlanggraph-langsmith/) |
| [`ECE469_Artificial_Intelligence/`](ECE469_Artificial_Intelligence) | Coursework for ECE-469 Artificial Intelligence at Cooper Union — homeworks (`hw1`–`hw3`), lecture slides, and two programming projects (a game-playing AI checker, an artificial neural network) | — (university course) |
| [`quantum_kernel_classification/`](quantum_kernel_classification) | Independent research project/paper: "When Do Quantum Kernels Help?" — a from-scratch NumPy simulation of the Havlíček et al. ZZ feature-map quantum kernel, benchmarked against classical kernels, with a NeurIPS-format writeup in `paper/` | — (independent research) |
| [`quantum_machine_learning_qiskit/`](quantum_machine_learning_qiskit) | "Quantum Machine Learning with Qiskit 2.x" course — Qiskit primitives (`SamplerV2`/`EstimatorV2`), classical-to-quantum data encoding/feature maps, variational quantum classifiers, and quantum-kernel SVMs, with notebooks numbered in build order (1 → 4c) and LaTeX notes on encoding schemes | [Quantum Machine Learning using Qiskit 2.x (Maven)](https://maven.com/faryad/aqml) |
| [`brownian_motion_sim/`](brownian_motion_sim) | C++ Brownian motion (Wiener process) simulator producing particle trajectory data | — (side project) |
| [`assets/`](assets) | Standalone images (e.g. STEM-GNN architecture diagrams) referenced from notes | — |

## Notes

- Some subfolders (`Deep_learning_cnns/machine_learning_examples`, `hyperparameter_optimization_for_ml/hyperparameter-optimization`, `quantum_machine_learning_qiskit`) are git submodules pointing at external course repositories — see [`.gitmodules`](.gitmodules).
- Course notes are largely written in LaTeX (`.tex`, compiled to `.pdf`) alongside Jupyter notebooks (`.ipynb`) and R scripts.
- `quantum_kernel_classification/` and `ECE469_Artificial_Intelligence/` aren't Udemy material — they're adjacent coursework/research kept here alongside it.
- `Deep_learning_cnns/` draws on Lazy Programmer's full course catalog (they share one `machine_learning_examples` repo), not a single course — the link above points to his course listing rather than one title. `quantum_machine_learning_qiskit/` is a Maven cohort course, not Udemy, despite living in this repo.
- This is a study/reference repo, not a packaged library — there's no single install step or entry point; open the relevant notebook, notes file, or project `README.md` for a given topic.
