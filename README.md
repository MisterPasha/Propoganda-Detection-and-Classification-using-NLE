## NLP Propaganda Detection & Classification

An academic Natural Language Processing (NLP) project investigating the identification and classification of propaganda techniques in text using traditional machine learning and transformer-based deep learning models.

The project explores two main tasks:

- **Propaganda Technique Classification:** Comparing One-vs-Rest Logistic Regression with Word2Vec/GloVe embeddings against an ALBERT-CNN architecture. The best transformer-based model achieved **60% classification accuracy**.
- **Propaganda Span Identification & Classification:** Identifying propaganda expressions within sentences using BIO sequence labelling. Compared BiLSTM-CRF with engineered linguistic features against fine-tuned ALBERT, achieving a best micro F1-score of **0.387**.

The research involved text preprocessing, feature engineering, transformer fine-tuning, hyperparameter optimisation and experimental evaluation, with particular attention to class imbalance, overfitting and model generalisation.

**Technologies:** Python, PyTorch, Hugging Face Transformers, scikit-learn, ALBERT, CNN, BiLSTM, CRF, Word2Vec, GloVe, TF-IDF.
