# 📊 Emotion Classification Model Results Summary

| **#** | **Model Type** | **Framework / Library** | **Dataset** | **Preprocessing Steps** | **Train / Val / Test Split** | **Features Used** | **Data Augmentation** | **Hyperparameters** | **Accuracy (Val)** | **Precision** | **Recall** | **F1 Score (Val)** | **Accuracy (Test)** | **F1 Score (Test)** | **Notes** | **Students Responsible** | **Evidence Links** |
|:--:|:--|:--|:--|:--|:--|:--|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--|:--|:--|
| 1 | Logistic Regression | scikit-learn | final_dataset.csv | features.py (baseline) | 80/0/20 | Sentiment / word2vec embeddings / bert embeddings | Sentiment reshaped, other features vertically stacked | max_iter = 1000 | 0.244 | 0.224 | 0.224 | 0.216 | – | – | [View Notes](#logistic-regression) | Filip | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 2 | Gaussian Naïve Bayes | scikit-learn | final_dataset.csv | features.py (baseline) | 80/0/20 | Sentiment / word2vec embeddings / bert embeddings | Sentiment reshaped, other features vertically stacked | var_smoothing = 0.0005 | 0.293 | 0.291 | 0.293 | 0.274 | – | – | [View Notes](#gaussian-naïve-bayes) | Filip | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 3 | Linear SVC | scikit-learn | final_dataset.csv | features.py (baseline) | 80/0/20 | Sentiment / word2vec embeddings / bert embeddings | Sentiment reshaped, other features vertically stacked | penalty='l2', loss='squared_hinge', C=500.0, multi_class='ovr', fit_intercept=True | 0.252 | 0.216 | 0.252 | 0.220 | – | – | [View Notes](#linear-svc) | Filip | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 4 | SimpleRNN-v0 | keras | final_dataset.csv – 1000 samples per class | features.py (baseline) + tf.keras.preprocessing.text.Tokenizer and tf.keras.preprocessing.sequence.pad_sequences | 80/0/20 | Sentence_Tok | None | One recurrent layer, lr=1e-4, kernel_regularizer="l2" for dense layers, batch_size=64, 0.4 dropout layer before classification head | 0.553 | 0.555 | 0.553 | 0.553 | – | – | [View Notes](#simplernn-v0) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 5 | SimpleRNN-v1 | keras | final_dataset.csv – 1000 samples per class | features.py (baseline) + tf.keras.preprocessing.text.Tokenizer and tf.keras.preprocessing.sequence.pad_sequences | 80/0/20 | Sentence_Tok | None | Two recurrent layers, lr=1e-4, kernel_regularizer="l2" for dense layers, batch_size=64, 0.4 dropout layer before classification head | 0.596 | 0.599 | 0.596 | 0.597 | – | – | [View Notes](#simplernn-v1) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 6 | LSTM-v0 | keras | final_dataset.csv – 1000 samples per class | features.py (baseline) + tf.keras.preprocessing.text.Tokenizer and tf.keras.preprocessing.sequence.pad_sequences | 80/0/20 | Sentence_Tok | None | One recurrent layer, lr=5e-3, kernel_regularizer="l2" for dense layers, batch_size=64 | 0.710 | 0.722 | 0.710 | 0.713 | – | – | [View Notes](#lstm-v0) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 7 | LSTM-v1 | keras | final_dataset.csv – 1000 samples per class | features.py (baseline) + tf.keras.preprocessing.text.Tokenizer and tf.keras.preprocessing.sequence.pad_sequences | 80/0/20 | Sentence_Tok | None | Two recurrent layers, lr=5e-3, kernel_regularizer="l2" for dense layers, batch_size=64 | 0.764 | 0.778 | 0.764 | 0.767 | – | – | [View Notes](#lstm-v1) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 8 | LSTM-v2 | keras | final_dataset.csv – 1000 samples per class | features.py (baseline) + tf.keras.preprocessing.text.Tokenizer and tf.keras.preprocessing.sequence.pad_sequences | 80/0/20 | Sentence_Tok, word2vec_embedding, bert_embedding | None | Two recurrent layers, lr=5e-3, kernel_regularizer="l2" for dense layers, batch_size=64, 0.4 dropout before dense layers | 0.791 | 0.803 | 0.791 | 0.793 | – | – | [View Notes](#lstm-v2) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 9 | LSTM-v3 | keras | final_dataset.csv – 1000 samples per class | features.py (baseline) + tf.keras.preprocessing.text.Tokenizer and tf.keras.preprocessing.sequence.pad_sequences | 80/0/20 | Sentence_Tok, bert_embedding | None | Two recurrent layers, lr=5e-3, kernel_regularizer="l2" for dense layers, batch_size=64, 0.4 dropout before dense layers | 0.796 | 0.806 | 0.796 | 0.799 | – | – | [View Notes](#lstm-v3) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 10 | DistilBERT | transformers | final_dataset.csv - 5000 samples per class | Tokenization using transformers tokenizer | 80/0/20 | Pretrained embeddings | None | Classification head modified, fine-tuned 3–5 epochs | 0.889 | 0.891 | 0.889 | 0.889 | 0.51 | 0.44 | [View Notes](#distilbert) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 11 | BERT-base-uncased | transformers | final_dataset.csv - 5000 samples per class | Tokenization using transformers tokenizer | 80/0/20 | Pretrained embeddings | None | Classification head modified, fine-tuned 5 epochs | 0.895 | 0.897 | 0.895 | 0.895 | 0.481 | 0.382 | [View Notes](#bert-base-uncased) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 12 | DistilRoBERTa base | transformers | final_dataset.csv - 5000 samples per class | Tokenization using transformers tokenizer | 80/0/20 | Pretrained embeddings | None | Classification head modified, fine-tuned 5 epochs | 0.901 | 0.904 | 0.901 | 0.901 | 0.482 | 0.435 | [View Notes](#distilroberta-base) | Razvan | [Notebook](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-12-1/blob/main/Notebooks/Model-Iterations.ipynb) |
| 13 | DistilBERT base | transformers | combined_emotion_dataset.csv - 10.000 samples per class | Tokenization using transformers tokenizer | 80/20/0 + show dataset | Pretrained embeddings | None | Classification head modified, fine-tuned 12 epochs (2 unfrozen layers) | - | - | - | - | 0.641 | 0.617 | [View Notes](#distilbert-fine) | Filip + Razvan | [Model Card](https://huggingface.co/dafaqboomduck/distilbert-sentiment-fine) |
| 14 | RoBERTa base | transformers | transcribed_data.csv | Tokenization using transformers tokenizer | 80/20/0 + show dataset | Pretrained embeddings | None | Classification head modified, fine-tuned 7 epochs (2 unfrozen layers) | - | - | - | - | 0.694 | 0.682 | [View Notes](#roberta-base) | Filip + Razvan |  |



---

## 🔍 **Results & Observations**

### Logistic Regression
<a name="logistic-regression"></a>

**Strengths:**  
Quick and simple model, can be used as baseline

**Weaknesses:**  
Very low scores,does not catch all emotions

**Other Notes:**  
This model is going to be used as a baseline for other models 

---

### Gaussian Naïve Bayes
<a name="gaussian-naïve-bayes"></a>

**Strengths:**  
- Overall best performing among these models

**Weaknesses:**  
 

**Other Notes:**  


---

### Linear SVC
<a name="linear-svc"></a>

**Strengths:**  
Good for separating  linear data

**Weaknesses:**  
Horrible at separating non-linear data

**Other Notes:**  
Worst performing of them all

---

### SimpleRNN-v0
<a name="simplernn-v0"></a>

**Strengths:**  
Relatively quick to run. Requires more resources compared to a Sklearn model, but less compared to an LSTM model or Transformer-based models. 

**Weaknesses:**  
Can't fully understand the linguistic patterns in the training data required for distinguishing between the 7 classes.

**Other Notes:**  
- Single recurrent layer

---

### SimpleRNN-v1
<a name="simplernn-v1"></a>

**Strengths:**  
Still relatively quick to run. It requires a bit more resources compared to the SimpleRNN-v0 model, which only had one recurrent layer, but it is still less computationally expensive compared to future models. 


**Weaknesses:**  
While somewhat better than the SimpleRNN-v0 model, it still can't fully understand the linguistic patterns in the training data required for distinguishing between the 7 classes.


**Other Notes:**  
- Two recurrent layers. 

---

### LSTM-v0
<a name="lstm-v0"></a>

**Strengths:**  
The LSTM layer is much better at understanding the linguistic patterns in the training data compared to the SimpleRNN layer or the Sklearn models. Its memory gates make it much more efficient at modelling the data. 

**Weaknesses:**  
Requires more computational power than the previously logged models. It takes more time to run. 

**Other Notes:**  
- Single LSTM layer.  

---

### LSTM-v1
<a name="lstm-v1"></a>

**Strengths:**  
The two LSTM layers improve the performance over the LSTM-v0 model. The model has a good modelling capability, being the first one so far to surpass the criteria for the F1 score. 

**Weaknesses:**  
Requires more computational power than the previously logged models. It takes more time to run. 

**Other Notes:**  
- Considered the base LSTM model for further improvements.  

---

### LSTM-v2
<a name="lstm-v2"></a>

**Strengths:**  
Uses the same architecture as the best-performing base model: LSTM-v1, but we added additional features as inputs, which slightly increased the model's performance.

**Weaknesses:**  
The added features make it require more computational power to run. The training time also slightly increases with the addition of these features. 

**Other Notes:**  
We believe that stacking these features might have introduced confusion for the model.

---

### LSTM-v3
<a name="lstm-v3"></a>

**Strengths:**  
Uses the same architecture as the best-performing base model: LSTM-v1. To address our concern with the previous model, we only added one feature: the BERT embeddings. This proved our assumption correct, as the results saw a slight increase compared to the last model. It also requires less computational power to run compared to the LSTM-v2 model

**Weaknesses:**  
Its performance increase didn't justify the effort put into it. Even though we added features that we assumed would increase its performance, the difference in accuracy and weighted F1 score was not as big as we expected.  
 
**Other Notes:**  
We believe that the performance increase starting to plateau means that we have reached or almost reached the peak performance of the current architecture, meaning that we will now move on to more powerful models, namely, Transformer-based ones, like BERT or ROBERTA.

---

### DistilBERT
<a name="distilbert"></a>

**Strengths:**  
We started off using a DistilBERT model, which is a lightweight version of the base BERT model. The specific model we ended up using was the "DistilBERT-base-uncased-sst-2-english" model, which is a finetuned model for emotion classification for English. The model was already trained on a large corpus of data, so we only had to change the classification head to match our desired output (6 classes + neutral) and finetune the model for a small number of layers until a satisfactory performance is reached. When evaluated on the validation set, the model showed a promising performance, outperforming the best model we had previously, the LSTM-v3 architecture. The weighted F1 score and the accuracy were both 88.9%. 

**Weaknesses:**  
The first downside of this model is that it takes a considerable amount of time to fine-tune. We only trained the model for a small number of epochs, between 3 and 5, and that took between 20 and 30 minutes for each run. The second downside is that it greatly overfits the training data. When evaluated on the validation set, which consists of a held-out percentage of the dataset used for finetuning, it achieves an incredible performance of about 89%. However, when we used our test set, the show dataset, the performance plummeted to 51% for accuracy and 44% for the weighted F1 score. This shows that the current approach causes the model to learn the hidden intrinsic patterns of the dataset used for training and perform very well when evaluated on a subset of it. However, when the test set is a new, never-seen dataset, the performance plummets as the model fails to generalize across multiple datasets well. 

**Other Notes:**  
- Shows the model learns dataset-specific patterns rather than generalizable emotion representations.  

---

### BERT-base-uncased
<a name="bert-base-uncased"></a>

**Strengths:**  
The second model we tried was the bert-base-uncased model. We tried this model as it has more parameters than the DistilBERT model we previously tried, and we believed that the increased number of parameters it has would help it better understand and create meaningful representations of the training data. We again changed the classification head to match our desired output (6 classes + neutral) and then we finetuned the model for a small number of epochs until a satisfactory performance was reached. We then used the held out evaluation set to test the performance of this new model and results were indeede better than the ones achieved by the DistilBERT model and much better than anything previously achieved. The model achieved an accuracy and weighted F1 score of 89.5%.
 
**Weaknesses:**  
The model took a significant amount of time to fine-tune. We only trained it for 5 epochs, and the training process took around 40-45 minutes to complete. Another major issue with this model is that its performance doesn’t extend across multiple datasets. When evaluated on the validation set, it achieved an excellent performance of nearly 90%. However, when we tested the model's performance on the show data, annotated by the company's pipeline, the results were far from our expectations. Although this model performed better during training and on the validation set compared to the previously evaluated DistilBERT model, it performed even worse on the test data. Its accuracy dropped to 48.1%, and the weighted F1 score decreased further, reaching 38.2%. This indicates that both models, despite performing well on the evaluation set, did not truly learn relevant linguistic patterns or significant, emotion-filled words during training and were only overfitting to that specific dataset. When tested on a different dataset, their true capabilities became evident. 

**Other Notes:**  
- Both DistilBERT and BERT fail to generalize, highlighting overfitting to validation data.  

---

### DistilROBERTA base
<a name="distilroberta-base"></a>

**Strengths:**  
The third model we tried was the DistilROBERTA base model. We tried this model because we wanted to leverage the power of the RoBERTa model family without having to deal with the computational requirements of a large model like the RoBERTa base or large versions. The model's training didn't take substantially more time than the two previously tried models which was good for us. The model achieved the best performance of all tested models so far, achieving scores of 90.1% for both accuracy and the F1 score. 
 
**Weaknesses:**  
When we tested the model's performance on the show data, annotated by the company's pipeline, the results were far from our expectations. Although this model performed better during training and on the validation set compared to the two previous models, it performed even worse on the test data. Its accuracy dropped to 48.2%, and the weighted F1 score decreased further, reaching 43.5%. 

**Other Notes:**  
- Having reached these resuls we decided to settle with this model and beging working on other tasks. 
- However, after seeing the resuls on the test set, it was quickly obvious this model is not what we though it was and we needed to train new, more capable models. 
- All three models, despite performing well on the evaluation set, did not truly learn relevant linguistic patterns or significant, emotion-filled words during training and were only overfitting to that specific dataset. We need to find a method of training models that generalizes well across different datasets. 

---

### DistilBERT fine
<a name="distilbert-fine"></a>

**Strengths:**  
- The model had a better generalization capability than the previously trained models who were only performing well because of the easy evaluation set used to test their performance. 
- The model performed very well on three classes: neutral, happiness, surprise.
- The model still uses the DistilBERT base so it is very lightweight compared to other transformer-based models, making inference faily quick to run. 
 
**Weaknesses:**  
- The model is still far from achieving the desired performance >= 75%. 
- Despite doing well on those three classes mentioned in strengths, it does very poorly for fear and disgust, with an F1 score of 35% for those two classes. 
- The model tends to default to neutral when unsure how to classify one sentence. 

**Other Notes:**  
- We will use this model for performing error analysis and XAI because of the time costraints of the project. 
- This model is the result of countless tries to improve the transformer-models and making them have a better performance accross different datasets. 
- The data we previously tested the models against was re-annotated to fix the mistakes made my the company's pipeline. 


---

### RoBERTa base
<a name="roberta-fine"></a>

**Strengths:**  
- The model was fine-tuned on a small dataset of data transcribed from other shows. This way its performance would be more closely aligned with the model's test set and intended use. We only unfreezed the last two layers in the model and swapped the classification head. This way we made training faster and more efficient, also ensuring the model doesn't forget the information he learned during its pretrining in the first layers. The model's perforamnce on show dataset was great, achieving an accuracy very close to 70% and a weighted F1 score of 68%. We also tested it on another test dataset and its performance was consistent accross the two. 
 
**Weaknesses:**  
- The model still doesn't perform as good for some classes as for others. It has a F1 score of 79% for neutral but only 37% for disgust. 
- It still classifies many sentences as neutral even when they aren't necessarily neutral. 

**Other Notes:**  
- The model almost achieved the desired performance with an F1 score of about 68%. Although this is not exactly what the target was, due to the time constraints of the project, we are satisfied with its peformance. 
- There is still room for improvement but there is no time left to do so. 
- We will use this model for the final presentation. We have to rerun the Error Analysis and XAI notebooks for this model. We will not remake the explanation documents but we will do this so during the presentation we can present any key findings about this new model. 


---

**🗓️ Document Prepared On:** 2025-10-26  
**🧑 Authors:** Filip, Razvan
