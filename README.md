# 588-project

This is a final project for the Winter 2024 semester of the University of Michigan class EECS 588: Computer Network and Security.

Our project investigates the use of continual learning and self-learning to address the concept drift present in spam data. We test using the Enron dataset, a common long-term spam/ham email dataset that importantly has timestamps labeled. The model implemented is a simple multi-layer perceptron that classifies off of RoBERTa encodings of email subject line and body content. Our model was able to beat the results of Guo, Mustafaoglu, and Koundals' 2023 paper "[Spam Detection Using Bidirectional Transformers and Machine Learning Classifier Algorithms](https://ojs.bonviewpress.com/index.php/JCCE/article/view/192)."

The more interesting contribution is our continual learning pipeline. We mimic a real world scenario where a company has a limited dataset of labeled spam/ham emails to do an initial training on, and then deploys a continually learning model that self-learns on predicted labels every time bucket. For a similar concept, see  [MORPH: Towards Automated Concept Drift
Adaptation for Malware Detection](https://www.ndss-symposium.org/wp-content/uploads/ndss24-posters-39.pdf).

We found that training a supervised model in this continual learning pipeline outperforms a model that's frozen after initial training, as expected. Our self-learning model did not perform well. There are many self-learning strategies we have not tried, so we believe this method could still be successful. Unsupervised learning that works as well as supervised learning would cut down on a great deal of resources, making it feasible to deploy a continual learning model that handles concept drift better than a frozen model.

Evaluation in Detail
------
We evaluated three training methods across three stages of evaluation. The Enron dataset is sorted temporally and divided into 5 even time chunks. The first two chunks represent the initial training set. In the first stage of evaluation, all three models undego supervised training on these first two chunks and are tested on the third chunk. This simulates being deployed after inital training and being tested on new emails as they arrive in real time. In the second stage of evaluation, the baseline model is frozen, the supervised "upper bound" is retrained on the data from the first three chunks with ground truth labels
, and our self-learning model retrains on the first two chunks with labels and the thrird chunk with its own predictions. This pattern continues until the dataset is exhausted. In our case, we chose to split the data into 5 chunks so there is only one more round of evaluation.
<img width="858" alt="Screenshot 2024-04-17 at 9 28 08 PM" src="https://github.com/julieshah24/588-project/assets/51678967/8771b43b-f899-4ee1-8df2-bc7d638cd2a9">
<img width="858" alt="Screenshot 2024-04-17 at 9 27 05 PM" src="https://github.com/julieshah24/588-project/assets/51678967/c26a24de-046a-4147-aaae-03375e72550c">
<img width="860" alt="Screenshot 2024-04-17 at 9 27 24 PM" src="https://github.com/julieshah24/588-project/assets/51678967/a4f5215d-5886-4cfa-b141-f7fc5a4989a9">

The results are shown below. As we achieved poor performance with our self-learning model, we introduced a fourth model that holds out a section of labeled data from the inital training to be introduced in later trainings. This model did better than the original self-learning model.
<img width="453" alt="Screenshot 2024-04-17 at 9 30 21 PM" src="https://github.com/julieshah24/588-project/assets/51678967/038e3dac-cacd-4c15-97e2-9a34c72b243a">

With the trained model weights, we statistically proved the existence of concept drift in the spam class of the Enron dataset that does not exist in the ham class. For each time chunk, we obtained the encodings of the email content using our trained RoBERTa weights, and reduced the dimensionality to 2 using t-SNE. We then used Laverne's test to show that the distributions had in-equal variances and the Shapio-Wilk test to show that distributions were not normal. From there, we used the Mann-Whitney U test to examine if there were statistically significant differences across the time chunks for each class. We found statistically significant differences across both dimesions of the spam class, but only across one dimension of the ham class. This proves that our spam class undergoes unique concept drift. This table summarizes the results of our statistical analysis.
<img width="805" alt="Screenshot 2024-04-17 at 9 38 52 PM" src="https://github.com/julieshah24/588-project/assets/51678967/3fde2d79-cedb-4eee-b2dd-a95e1cb298eb">

One note to these results is that the Enron dataset has only ham emails for the first two years and only spam emails for other three years. To handle this discrepancy, we did the chunking for each class individually and merged the classes for each chunk. This is reasonable due to the lack of drift in ham emails. 
