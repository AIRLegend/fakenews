# Fake news detection
------

This repository is focused on finding fake news using deep learning

There are multiple methods focused on achieving this goal, but the objective 
of this work is discriminating the fake ones by only looking at the text. No graphs,
no social network analysis neither images.

### Sources
1. Fake News Corpus: https://github.com/several27/FakeNewsCorpus
2. Getting Real About Fake News: https://www.kaggle.com/mrisdal/fake-news
3. Fake News Detection: https://www.kaggle.com/jruvika/fake-news-detection

### Folder structure
* **data**: This directory must be created with the necessary data for scripts to work.
            (Not uploaded to GH due to filesize restrictions).
  - GoogleNews-vectors-negative300.bin.gz
  - Other_datasets
        - GettingRealAboutFake/
        - fake-news-detection/
        - real_or_fake.csv
  - FakeNewsCorpus.csv
* **notebooks**: Notebooks for prototyping
* **src**: Code with utils
  * **data**: Code to generate datasets / process data


### Notebook explanation
* *FakeNewsCorpus.ipynb:* Cleaning and preprocessing the dataset 'Fake News Corpus'.
* *GettingRealAboutFake.ipynb:* Cleaning and preprocessing the dataset 'Getting Real
  About Fake News' from Kaggle.
* *Processing_test_dataset.ipynb:* Cleaning and preprocessing the dataset 'True or Fake' from Kaggle.
* *BayesianOpt.pynb:* Obtaining model hyperparameters using Bayesian Optimization
* *Train-Colab-Categorical.ipynb:* Train DNN to categorize 4 types of news.
* *Train_Colab_Binary.ipynb:* Train DNN to categorize only **True** or **Fake**
  classes.
* *Test_Colab_Categorical.ipynb:* Testing the previously trained categorical models.
* *Test_Colab_Binary.ipynb:* Testing the previously trained binary models.
