*************************** ATE ABSITA - EVALITA 2020 **************************

Contacts:
  website: http://www.di.uniba.it/~swap/ate_absita/index.html
  email: ate.absita.evalita2020@gmail.com

PLEASE CITE:

@inproceedings{demattei2020overview,
  title={{Overview of the evalita 2020 ATE\_ABSITA: Aspect Term Extraction and Aspect-basedSentiment Analysis task}},
  author={de Mattei, Lorenzo and De Martino, Graziella and Iovine, Andrea and Miaschi, Alessio and Marco, Polignano},
  booktitle={EVALITA 2020-Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian},
  year={2020},
  organization={CEUR}
}


***************************** TASK DESCRIPTION *********************************

In our challenge we would like to propose three different annotation tasks regarding
Aspect Term Extraction (ATE), Aspect Based Sentiment Analysis (ABSA) and sentence
Sentiment Analysis (SA).

Aspect Term Extraction (ATE) is the task of identifying an "aspect" in a text
without knowing a priori the list that contains it. According to the literature
definition, a term/phrase is considered as an aspect when it co-occurs with
“opinion words” that indicate a sentiment polarity on it.
More details and examples are available at:
http://www.di.uniba.it/~swap/ate_absita/examples.html

Aspect-based Sentiment Analysis (ABSA) is an evolution of Sentiment Analysis that
aims at capturing the aspect-level opinions expressed in natural language texts.
In the Aspect-based Sentiment Analysis (ABSA) task, the polarity of each expressed
aspect is recognized.

Sentiment Analysis (or Opinion Mining) is the task of identifying what the user
thinks about a particular piece of text. In particular, it often takes the form
of an annotation task with the purpose of annotating a portion of text with a
positive, negative, or neutral label.
In our Sentiment Analysis (SA) task, the polarity of the review is provided.
In particular we decided to use the score left by the user at the item as
value of polarity. It is defined as an integer number into the range 1:5.

***************************** DATASET DETAILS *********************************

We have collected 4364 real-life user reviews, written in the Italian language,
about 23 products. The training, dev and test sets is randomly generated in the
portion: 70% training, 2.5% dev, 27.5% test set.
This mean that the test set will be not out-of-domain. The data format used is
NDJSON (http://ndjson.org/) with UTF-8 encoding and newline as delimiter.
Note that some reviews may not contain any aspect, but the final review score
is always available.

DEV SET: 109 reviews - ate_absita_dev.ndjson - 37 KB
TRAINING SET: 3054 reviews - ate_absita_training.ndjson - 1.1 MB
TEST SET: 1200 reviews - ate_absita_test.ndjson - 322 KB

***************************** RELEASE POLICY ***********************************

All material used and produced by the organizers of this evaluation task is
released for non-commercial research purposes only. In this regard, no tools
are provided to link the reviews released as datasets, to specific subjects on
the web, or to trademarks and third parties. Furthermore, any use for statistical,
propagandistic or advertising purposes of any kind is prohibited. It is not
possible to modify, alter or enrich the data provided for the purposes of
redistribution.

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
http://creativecommons.org/licenses/by-nc-nd/4.0/

********************************************************************************
