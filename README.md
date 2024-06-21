#  Classifying-hand-written-letters

## 🤖 Supervised image classification on the EMNIST dataset. 

- Created training and testing subsets to train several machine learning models.
- Perform inference on the testing data 
- Results are then evaluated and compared to determine the most suitable machine learning model.

## 📊 Dataset 
- The EMNIST dataset is contained in the file _dataset-letters.mat_ which contains a variable _dataset_ holding images and labels.
- Each image is of size 28 x 28 pixels, stored as a reshaped 1 x 784 vector.

## 🧰 Data Preparation
- Data is loaded onto _MATLAB_ and the images are converted to a double data type.
- The dataset was then divided equally, allocating 50% for
training—to learn and identify patterns—and the other 50% for
testing to evaluate the models' predictive accuracy.

## 🏗️ Methodology 
- The K-Nearest Neighbours (KNN) algorithm was selected for its
straightforward approach to predicting outcomes by looking at
similar instances, using different ways to measure similarity.
-  Support Vector Machine (SVM) was included for
its capability to handle complex relationships in data, and Decision Trees were used for their ease of
interpretation and decision-making transparency.
- These models were trained on the EMNIST data and was
tested to gauge their predictive accuracy.
- To evaluate the performance of our models, accuracy was used,
which simply measures the percentage of predictions our models got right. This approach helped to
determine which model was most effective in correctly classifying the data.

## 👨‍🏫 Results 
- The k-nearest neighbours algorithm utilizing the Euclidean distance metric outperformed others with
an accuracy of 78.86%
-  Despite its longer prediction time of 33.76 seconds, the high accuracy rate positions
it as the recommended model for situations where precision is paramount.
- The Support Vector Machine, with an accuracy of 73.52%,
stands out for its quicker prediction time of 4.04 seconds, presenting a viable alternative when faster
prediction is required without a substantial sacrifice in accuracy.
- The Decision Tree model lagged behind
with a notably lower accuracy of 56.46%, although it had the fastest prediction time of 0.02 seconds. This
trade-off suggests its suitability for applications where speed is critical and some loss in accuracy is
tolerable
