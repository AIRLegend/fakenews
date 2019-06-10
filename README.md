# Fake news detection using deep learning
## Final master thesis project

This repository is focused on finding fake news using deep learning

There are multiple methods focused on achieving this goal, but the objective 
of this work is discriminating the fake ones by only looking at the text. No graphs,
no social network analysis neither images.

In this work three deep learning architectures are proposed and later tested over two datasets (Fake news corpus and TI-CNN), obtaining state of the art results.

1. **LSTM Based architecture**: $91\%$ accuracy (TI-CNN) ||  $76\%$ accuracy (FNC)
2. **CNN Based architecture**: $97\%$ accuracy (TI-CNN) || $82\%$ accuracy (FNC)
3. **BERT Based architecture**: $97\%$ accuracy (TI-CNN) || $76\%$ accuracy (FNC)

This repository contains several Python notebooks with the developed code 

### Data sources
* Fake News Corpus: https://github.com/several27/FakeNewsCorpus
* Getting Real About Fake News: https://www.kaggle.com/mrisdal/fake-news
* Fake News Detection: https://www.kaggle.com/jruvika/fake-news-detection
* News Dataset from TI-CNN: https://arxiv.org/abs/1806.00749

### Folder structure
* **data**: This directory must be created with the necessary data for scripts to work.
            (Not uploaded to GH due to filesize restrictions).
  - GoogleNews-vectors-negative300.bin.gz: Word2Vec news trained model weights
  - Other_datasets
    - GettingRealAboutFake/
    	- ```fake.csv``` (*Getting Real About Fake News Dataset*)
    	- ``all_data.csv`` (*TI-CNN dataset*) 	
    - ``real_or_fake.csv``
  - `FakeNewsCorpus.csv` (*Fake News Corpus*)
* **notebooks**: Notebooks for prototyping
* **src**: Code with utils
  * **data**: Code to generate datasets / process data
  * **bert_class**: Fine-tuned classifier built over  Google's BERT to detect fake/true news.

### Notebook explanation
* `FakeNewsCorpus.ipynb:` Cleaning and preprocessing the dataset 'Fake News Corpus'.
* `GettingRealAboutFake.ipynb:` Cleaning and preprocessing the dataset 'Getting Real
  About Fake News' from Kaggle.
* `TI_CNN-Dataset:` Cleaning and preprocessing the tadaset 'TI-CNN'.
* `Processing_test_dataset.ipynb:` Cleaning and preprocessing the dataset 'True or Fake' from Kaggle.
* `BayesianOpt.pynb:**` Obtaining model hyperparameters using Bayesian Optimization
* `Train-Colab-Categorical.ipynb:` Train DNN to categorize 4 types of news.
* `Train\_Colab_Binary.ipynb:` Train DNN to categorize only **True** or **Fake**
  classes.
* `Test\_Colab_Categorical.ipynb:` Testing the previously trained categorical models on TI-CNN.
* `Test\_Colab_Binary.ipynb:` Testing the previously trained binary models on FNC.
* `data\_analysis/Data_analysis-FNC.ipynb:` EDA of the Fake News Corpus.
* `data\_analysis/Data_analysis-TI-CNN.ipynb:` EDA of the TI-CNN Dataset
* `data\_analysis/Data_analysis-Getting-Real.ipynb:` EDA of the Getting Real About
  fake news Dataset

