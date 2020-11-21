# Sentiment analysis on product user reviews

**Executive Summary**

Sentiment Analysis (or Opinion Mining) is the task of identifying what the user thinks about a particular piece of text. Sentiment analysis often takes the form of an annotation task with the purpose of annotating a portion of text with a positive, negative, or neutral label.

In this Sentiment Analysis (SA) task, the presented deep learning regression model is able to predict the score assigned by a user in a product review.

This task has been completed to participate to ATE_ABSITA at EVALITA 2020: http://www.di.uniba.it/~swap/ate_absita/task.html#

**Feature set information**

4364 real-life product user reviews, written in the Italian language, about 23 products. The training, dev and test sets is randomly generated in the
portion: 70% training, 2.5% dev, 27.5% test set.
This mean that the test set will be not out-of-domain. The data format used is NDJSON (http://ndjson.org/) with UTF-8 encoding and newline as delimiter.
Note that some reviews may not contain any aspect, but the final review score is always available.

- TRAINING SET: 3054 reviews - ate_absita_training.ndjson - 1.1 MB
- DEV SET: 109 reviews - ate_absita_dev.ndjson - 37 KB
- TEST SET: 1200 reviews - ate_absita_test.ndjson - 322 KB - *NOT YET RELEASED*

*Feature example*
```
{
    sentence: "Ottimo rasoio dal semplice utilizzo. Rade molto bene e in qualsiasi direzione. Pratico e facile da pulire"
    score: 5
}
```
**Data augmentation**

Additional reviews have been downloaded to enrich the training set. Those reviews are stored in the ```additional_scraped_reviews``` folder. Each file includes the reviews of one product.

**How this works**

The dataset is modeled using the following approach:

1) ```dataframe_pipeline.py``` converts the ndjson input file into a pandas dataframe that is then saved into the joblib_not_processed_dataframe folder in joblib format;
2) ```additional_features_preprocessing.py``` works on the product reviews added later to this dataset and stored in the ```additional_scraped_reviews``` folder;
3) ```preprocessing.py``` loads the dataset created in step 1 and applies the first cleaning layer on the data by removing i) punctuation, ii) numbers, iii) single characters, iv) multiple spaces and V) stopwords. After the cleaning layer is completed, the output is again saved in joblib format in the joblib_processed_features folder;
4) ```train.py``` makes the last part of preprocessing including applying word embeddings to create the feature matrices. The model is then saved into the ```models``` folder. To run this, you need to download the word embeddings from https://fasttext.cc/docs/en/crawl-vectors.html, create an ```embeddings``` folder into the main directory of the project and place the file downloaded within it. The expected structure of the directory/folder is:

```
embeddings
--cc.it.300.vec
```

**Model used and metrics**

RMS error on the dev set with the current model and data augmentation is 0.9978965872313215 (competition baseline: 1.004)

![](https://github.com/marcogdepinto/Sentiment_Analysis_On_Product_User_Reviews/blob/master/models/model.png)

![](https://github.com/marcogdepinto/Sentiment_Analysis_On_Product_User_Reviews/blob/master/loss.png)

**License**

All material used and produced by the organizers and the researcher for this evaluation task is released for non-commercial research purposes only. In this regard, no tools are provided to link the reviews released as datasets, to specific subjects on the web, or to trademarks and third parties. Furthermore, any use for statistical, propagandistic or advertising purposes of any kind is prohibited. It is not possible to modify, alter or enrich the data provided for the purposes of redistribution.

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License: http://creativecommons.org/licenses/by-nc-nd/4.0/

**EVALITA Credits**

- Website: http://www.di.uniba.it/~swap/ate_absita/index.html
- Email: ate.absita.evalita2020@gmail.com

```
@inproceedings{demattei2020overview,
  title={{Overview of the evalita 2020 ATE\_ABSITA: Aspect Term Extraction and Aspect-basedSentiment Analysis task}},
  author={de Mattei, Lorenzo and De Martino, Graziella and Iovine, Andrea and Miaschi, Alessio and Marco, Polignano},
  booktitle={EVALITA 2020-Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian},
  year={2020},
  organization={CEUR}
}
```
