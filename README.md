# Effect-of-heterogeneity-and-cluster-analysis
Analyze the effect of heterogeneity in medical images on the clustering of image features extracted from CNN

## Purpose: 
To assess whether the heterogeneity in the medical images affects the clustering of image features extracted from the CNN model.


## Research Question: 
Will the CNN models retrained from the medical images in each cluster obtained from the clustering of image features extracted from CNN,  perform differently?


## Objectives:
- Compare the test accuracy of the overall binary CNN model in the clusters assigned in the testing set. 
- Describe the re-training procedure of the binary CNN models from the medical images in each cluster obtained from the clustering of image features extracted from CNN.
- Compare the retrained CNN models’ performances on the testing set to see whether they differ.
- Explain how the heterogeneity in the dataset affects the clustering of image features

## Methods:

### Data:

Fundus Images (Cataract vs. Normal) Dataset

Chest X-Ray (Cardiomegaly vs. Normal) Dataset

To be added: [Other datasets from John Valen]

### Binary Classification CNN Model:  

We used the Keras library, a Python-based high-level neural network API, to build and train a binary classification model based on the EfficientNetB0 architecture, a state-of-the-art convolutional neural network that has demonstrated strong performance on image classification tasks. The pre-trained EfficientNetB0 model, which was trained on the large-scale ImageNet dataset, was used as a starting point. To fine-tune the model for our specific binary classification task, we added a custom output layer consisting of a single neuron with a sigmoid activation function. During training, the pre-trained layers were frozen, and only the output layer was trained using the binary cross-entropy loss function and the Adam optimizer. Early stopping and model checkpoint techniques were used to prevent overfitting. We did not use cross-fold validation since we utilized all training data for training the model.
To classify cataract vs. normal (Fundus Images), the model was trained for 10 epochs with a batch size of 8 on the training set, while monitoring its performance on the validation set. EarlyStopping was applied when the validation accuracy did not improve for three consecutive epochs. To classify cardiomegaly vs. normal (Chest X-Ray Images), the model was trained for 30 epochs with a batch size of 8 while monitoring its performance on the validation set. EarlyStopping was applied when the validation accuracy did not improve for 10 consecutive epochs, as this model is more difficult to train than the model to classify cataract and normal.

### Extract image features from the medical images: 
To extract features from the medical images, we removed the last layer of the model and created a new model that outputs the activations of the second last layer. We used this new model as a feature extractor to transform the input images into a lower-dimensional space of extracted features. Both the features in the images of the training set and testing set were extracted by the CNN model.

### Cluster analysis on the image features extracted from the CNN: 
We conducted cluster analysis on image features extracted from a CNN. Since the extracted features have high dimensionality, we used principal component analysis (PCA) to reduce the dimensionality and applied it to the training and testing data of fundus images. We selected the first 200 principal components to explain over 70% of the variance. For the Chest X-Ray Dataset, the test set only have 57 images, selecting the maximum number of components (57) could only explain 40% variance for the features in the training set, we actually compared two approaches to do the cluster analysis, in the first approach, we do the cluster analysis directly based on the image features extracted from CNN, in the second approach, we do the cluster analysis based on the 57 principal components. We then used the K-means clustering algorithm to group images with similar characteristics into four clusters, following the previous work from [4]. The K-means algorithm is preferred due to its computational efficiency, scalability, and simplicity. We assigned the transformed testing set into the four clusters based on the cluster centers found from the training set. For each cluster assigned in the testing set, we calculated the accuracy of the binary CNN classification model trained above.

### Retrain the binary classifiers on each of the clusters in the training set: 
To retrain the model on the clusters, we first obtained the cluster labels for each sample in the dataset using the K-means algorithm. We then created separate data frames for each cluster containing the image features of the images belonging to the corresponding cluster. Each model was then trained using the images belonging to the corresponding cluster. We used the same CNN architecture and hyperparameters that were used in the initial binary classification model training. However, we only used the samples belonging to the corresponding cluster for training each model.

### Train binary classifiers on random medical images with the same size as each cluster: 
To ensure a fair comparison of the performances of the retrained models from each cluster, we also trained a binary classification model on random medical images from the training set with the same size as each of the four clusters. This is to prevent any bias introduced by the varying sample sizes in each cluster when comparing their performances. We randomly sampled the same number of images as each cluster from the training set, created four separate data frames, and trained a binary classification model on each of them. To obtain more reliable estimates of the model's performance, we trained the model five times for each sample size and averaged the performance metrics over the five runs.


## Experiments: 

- Compare the test accuracy of the CNN model in the clusters assigned in the testing set: For both the Fundus Images Dataset and the Chest X-Ray dataset, we tested the performance of the binary classifier trained from the data in the training set on the four clusters assigned in the testing set and compared the testing accuracy of the model on the test clusters. 

- Compare the performances of the models retrained from each cluster: For both the Fundus Images Dataset and the Chest X-Ray dataset, the performance metrics (Test Accuracy, AUC, Sensitivity, Specificity) of the models retrained from each cluster were compared. As the cluster sizes vary, we also compared the retrained models with the performance of models trained from random medical images with training size equivalent to their corresponding cluster.


## Statistics and Outcome Measures: 

The main outcome measure in our study is the model testing accuracy, which is the proportion of the total number of corrected classified samples among the total number of samples in the testing set. Additionally, AUC, Specificity and Sensitivity will also be taken into account when comparing the model performances. We measured the between-cluster distance to observe how distant clusters are from each other. The between-cluster distance is calculated based on the distances between the centroids of each cluster. We used intra-cluster distance to measure the similarity of the images in each cluster, the intra-cluster distance is a measure of how close the data points are to each other within a cluster. It is calculated as the average distance between all pairs of data points within the cluster. Clusters with smaller intra-cluster distances are considered to be more compact and homogeneous, while clusters with larger intra-cluster distances are less compact and less homogeneous [7].

## Environment: 
The experiments are done using Google Colab, and the library used for training the CNN model is Keras.

