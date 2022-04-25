# NYCU-intro-to-AI-final-project 交大資工人工智慧概論 期末專題

## Problem Statement
Our goal is, for aerial images and environmental features of GPS positions in the test set, to return a set of candidate pecies that should contain the true observed species. The model is aimed to predict the species which are most likely to be observed at a given location. This is useful for many scenarios related to biodiversity management and onservation. More details can be found in this [kaggle](https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/overview) page.

## Description
There are large number of factors pertaining to the habitability of certain species. Furthermore, the given information, aerial images and rough environmental featrues, may not be sufficient to precisely determine the local species.

## I/O
![](https://raw.githubusercontent.com/maximiliense/GLC/master/images/patches_sample_FR.jpg)

The output is a two-column csv. The first column is the id of the location, and the second is the id of the top-30 species that are most likely to appear in that location.

## Related Work
The winner of GeoLifeCLEF 2021: http://ceur-ws.org/Vol-2936/paper-140.pdf

The winner of GeoLifeCLEF 2020: https://hal.inria.fr/hal-02989084/file/paper_192.pdf

## Methology
We would probably take a CNN approach or self-attention approach to this problem. The CNN is well-known for image-related problem, such as image classification. And recently self-attention is popping up on the field of computer vision, which we would like to utilize in our problem. While self-attention is a border class of CNN, the required training time and resources is also higher than CNN. We may also use random forest on this problem, expecting to get a better performance.


## Evaluation Metrics

We use top-30 accuracy as our evaluation metric. The accuracy is computed as below:

![](https://i.imgur.com/wTSEYpF.png)

The performance evalution is intuitive and natural. Because there may exist 
numerous species in the same area, top-30 evaluation is appropriate, not too demanding and not too loose.

## Baseline
Random selecting model is the lower bound of a model’s performance. Any reasonably well performed model should be able to surpass this baseline quite easily. Also, we use random forest, a traditional appreach, to be a medium baseline to examinate a model.

## Time Schedule
- 4/19-5/3 baseline model implementation.
- 5/4-5/24 try differnet models and differnet architectures and test performance.
- 5/25- deadline write report
