# Twitter Entity Sentiment Analysis — EDA, Classical ML & BERT

This repository contains an end-to-end sentiment analysis study on the **Twitter Entity Sentiment Analysis** dataset.

The main goal was not just to train one classifier, but to first understand the dataset carefully, then compare several classical NLP models under the same evaluation setup, and finally test a transformer-based model.

The notebook includes:

- data quality checks,
- exploratory data analysis,
- text and entity-level analysis,
- TF-IDF feature engineering,
- multiple classical machine-learning models,
- error analysis,
- optional hyperparameter tuning,
- BERT fine-tuning,
- and a final comparison of all models.

---

## Dataset

The dataset contains tweets associated with different entities such as companies, games, and technology products.

Each sample contains:

- `TweetID`
- `Entity`
- `Sentiment`
- `Text`

The sentiment labels are:

- `Positive`
- `Negative`
- `Neutral`
- `Irrelevant`

### Initial dataset size

| Split | Samples |
|---|---:|
| Training | 74,682 |
| Validation | 1,000 |

The training set contained:

- **686 missing text values**
- **2,700 exact duplicate rows**

After removing missing text rows and exact duplicates, the training set was reduced to:

**71,484 samples**

The dataset contains **32 unique entities**.

---

## Data quality observation

One important finding from the analysis is that the official training and validation files are not completely independent at the text level.

There are:

**516 exact tweet texts shared between training and validation**

which corresponds to approximately:

**51.65% of the unique validation texts**

This is important when interpreting the final validation scores. Very high validation accuracy should therefore not automatically be interpreted as equivalent performance on a completely unseen real-world dataset.

---

## Exploratory Data Analysis

Before training the models, I examined the dataset from several angles.

### Sentiment distribution

The cleaned training set contains:

| Sentiment | Samples | Percentage |
|---|---:|---:|
| Negative | 21,652 | 30.29% |
| Positive | 19,677 | 27.53% |
| Neutral | 17,651 | 24.69% |
| Irrelevant | 12,504 | 17.49% |

The classes are not perfectly balanced, but none of them is extremely rare.

### Entity distribution

There are **32 entities**, with roughly similar numbers of samples per entity.

The number of tweets per entity ranges from approximately:

- **2,139** samples
- to **2,318** samples

This makes the entity distribution relatively balanced.

### Entity-specific sentiment

The analysis also shows that different entities have noticeably different sentiment profiles.

Examples:

- **MaddenNFL** has a particularly high proportion of negative tweets.
- **Amazon** has a large neutral share.
- **PUBG**, **Battlefield**, and **Fortnite** have relatively high proportions of irrelevant tweets.

This suggests that the entity itself may provide useful information for sentiment prediction.

---

## Text analysis

Several text-level features were explored:

- character count,
- word count,
- URL frequency,
- mention frequency,
- hashtag frequency,
- question marks,
- exclamation marks.

Average tweet length also varies across sentiment classes.

For example, positive tweets are somewhat shorter on average than neutral and negative tweets in this dataset.

The notebook also analyzes:

- most frequent words,
- most frequent bigrams,
- vocabulary differences across sentiment classes.

---

## Preprocessing

Text preprocessing was intentionally kept conservative.

The pipeline:

- converts text to lowercase,
- replaces URLs with a `URL` token,
- replaces mentions with a `USER` token,
- normalizes whitespace.

Aggressive stop-word removal was avoided because words such as **not**, **no**, and **never** can be important for sentiment classification.

---

## Feature Engineering

The classical machine-learning models use two sources of information.

### 1. Tweet text

Tweet text is converted into TF-IDF features using:

- unigrams,
- bigrams,
- sublinear term frequency,
- up to 40,000 text features.

### 2. Entity

The entity column is one-hot encoded and combined with the TF-IDF representation.

The final feature matrix contains:

**40,032 features**

This allows the model to use both the linguistic content of the tweet and the entity being discussed.

---

## Models

The following classical models were evaluated using the same feature representation and validation set:

- Dummy Classifier
- Complement Naive Bayes
- Logistic Regression
- Linear SVM
- SGD Classifier
- Ridge Classifier
- Decision Tree

A BERT model was also fine-tuned separately.

---

## Classical ML Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **Linear SVM** | **0.978** | **0.9788** | **0.9780** |
| Ridge Classifier | 0.976 | 0.9761 | 0.9761 |
| Logistic Regression | 0.969 | 0.9687 | 0.9690 |
| SGD Classifier | 0.956 | 0.9551 | 0.9559 |
| Complement Naive Bayes | 0.881 | 0.8802 | 0.8808 |
| Decision Tree | 0.789 | 0.7884 | 0.7906 |
| Dummy Baseline | 0.266 | 0.1051 | 0.1118 |

The strongest classical model was **Linear SVM** with:

- **97.8% validation accuracy**
- **97.88% macro F1**

Its class-level performance was:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Irrelevant | 0.99 | 0.98 | 0.99 |
| Negative | 0.98 | 0.98 | 0.98 |
| Neutral | 0.98 | 0.97 | 0.98 |
| Positive | 0.96 | 0.99 | 0.97 |

Only **22 out of 1,000 validation samples** were misclassified.

---

## Error Analysis

The most common classification error was:

**Neutral → Positive**

with 6 cases.

Other frequent confusions included:

- Negative → Positive
- Positive → Neutral
- Positive → Negative
- Negative → Neutral

The notebook also measures performance separately across entities.

Some entities had perfect validation accuracy in this split, while others were more difficult.

Among the lower-performing entities were:

- Facebook
- PlayStation5
- Grand Theft Auto
- Apex Legends
- PUBG
- Red Dead Redemption

This analysis is useful because overall accuracy alone can hide entity-specific weaknesses.

---

## BERT Experiment

The transformer experiment uses:

**`bert-base-uncased`**

The input is constructed as:

```text
Entity: <entity>. Tweet: <tweet text>
```

so that the model receives both the entity and the tweet.

For the current experiment:

- training subset: **30,000 samples**
- validation set: **1,000 samples**
- epochs: **2**
- maximum sequence length: **128**
- learning rate: **2e-5**

### BERT results

| Metric | Score |
|---|---:|
| Accuracy | 0.799 |
| Macro F1 | 0.7986 |
| Weighted F1 | 0.7977 |

Class-level results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Irrelevant | 0.81 | 0.79 | 0.80 |
| Negative | 0.81 | 0.86 | 0.84 |
| Neutral | 0.80 | 0.70 | 0.75 |
| Positive | 0.78 | 0.84 | 0.81 |

In this experiment, BERT did **not** outperform the strongest TF-IDF-based linear models.

This should not be interpreted as a general conclusion that linear models outperform transformers for sentiment analysis. The BERT experiment used only a **30k subset** of the training data and only **2 epochs**, while the classical models were trained on the full cleaned training set.

The train/validation text overlap also makes the very high classical validation results worth interpreting cautiously.

---

## Final Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| **Linear SVM** | **0.978** | **0.9788** |
| Ridge Classifier | 0.976 | 0.9761 |
| Logistic Regression | 0.969 | 0.9687 |
| SGD Classifier | 0.956 | 0.9551 |
| Complement Naive Bayes | 0.881 | 0.8802 |
| BERT | 0.799 | 0.7986 |
| Decision Tree | 0.789 | 0.7884 |
| Dummy Baseline | 0.266 | 0.1051 |

For this particular experimental setup, **Linear SVM with TF-IDF and entity features provided the strongest validation result**.

---

## Key Takeaways

A few things stood out during this project:

1. **Strong linear baselines are extremely competitive for sparse NLP problems.**  
   Linear SVM and Ridge performed very well with TF-IDF features.

2. **Entity information matters.**  
   Sentiment distributions differ substantially across entities, so including the entity as a feature is reasonable for this task.

3. **Accuracy alone is not enough.**  
   Macro F1, confusion matrices, entity-specific accuracy, and manual error inspection give a much better picture of model behavior.

4. **Dataset leakage needs to be checked before celebrating a high score.**  
   More than half of the unique validation texts appear exactly in the training set, which likely contributes to the unusually high classical-model performance.

5. **A transformer is not automatically better in every experimental setup.**  
   The BERT run was intentionally limited in training size and epochs, so it should be treated as a first transformer baseline rather than a fully optimized comparison.

---

## Project Structure

```text
.
├── eda-machine-learning.ipynb
└── README.md
```

---

## Main Libraries

```text
pandas
numpy
matplotlib
scikit-learn
transformers
datasets
torch
```

---

## Running the Notebook

The notebook was designed for **Kaggle**.

For the classical ML sections, a GPU is not required.

For BERT fine-tuning:

1. Open the notebook settings.
2. Enable a GPU accelerator.
3. Make sure the Hugging Face model can be downloaded.
4. Set:

```python
RUN_BERT = True
```

Then run the BERT section.

---

## Possible Improvements

Some experiments I would like to try next:

- character-level TF-IDF,
- combined word + character TF-IDF,
- hyperparameter tuning for Linear SVM and Logistic Regression,
- more rigorous duplicate and near-duplicate handling,
- a new train/test split grouped by normalized tweet text,
- training BERT on the full cleaned training set,
- DistilBERT and RoBERTa comparisons,
- model ensembling,
- coefficient-based interpretability for the linear models,
- SHAP-based analysis,
- probability calibration.

A particularly important next step would be to rebuild the evaluation split after removing exact and near-duplicate text overlap. That would provide a more realistic estimate of generalization performance.

---

## Notebook

The full analysis, visualizations, model training, error analysis, and BERT experiment are available in:

```text
eda-machine-learning.ipynb
```
