# Intro

Description: This file is a summary of sklearn official tutorial.

---

## Basic Concept

Machine learning problem is to generate a predicting model based on the data set. Usually, we need to process a m $\times$ n matrix. Each column stand for a different preditors, attributes or features and each entry stand for an observation of the attributes.

---

## Classification Methods

- Supervised learning
  
  Supervised learning is appliable to the data set with addtional attributes we need. For example, for a problem requiring us to classify the character. If the data set contains addtional attributes like a catalog of the character each row corresponding to, we can apply the supervised learning.

  Supervised learning can be divided into 2 classed: classification and regression

  - Classification

    If the target is to assign the input into corresponding discrete catalog, we should use the classification.

  - Regression

    If the result should be a continuous function, we should use regression.

- Unsupervised learning

  If the data set doesn't contain the corresponding catalog each row corresponds to, we should use the unsupervised learning to establish the model. The principle of the unsupervised learning is to discover groups of similar examples within the data, where it is called clustering, or to determine the distribution of data within the input space, known as density estimation, or to project the data from a high-dimensional space down to two or three dimensions for the purpose of visualization.

---

## Training Set and Testing Set

To avoid the overfitting, ussualy, we divide the data set into training set and testing set. The trainging set is used to establish the model while the testing set is used to evaluate the quality of the model.
