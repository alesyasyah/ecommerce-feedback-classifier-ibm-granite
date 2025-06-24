# E-commerce Feedback Classifier using IBM Granite

## Project Overview

### Background
In the competitive e-commerce landscape, understanding customer feedback is crucial for business success. Companies receive thousands of reviews daily, containing valuable insights into customer satisfaction, product quality, and service issues. However, manually analyzing this vast amount of text is inefficient, slow, and does not scale.

### Problem Statement
The business problem is the inability to efficiently process and understand customer feedback at scale. This leads to a slow reaction time to critical issues (like poor product quality or shipping problems) and missed opportunities to capitalize on what customers love. The key questions are:
* How can we automate the analysis of customer feedback to quickly gauge sentiment?
* How can we identify the main drivers of positive and negative customer experiences?

### Project Goal
The goal of this project is to develop a proof-of-concept AI model that automatically classifies e-commerce customer reviews into 'Positive', 'Neutral', or 'Negative' categories. This will demonstrate a scalable solution for rapidly extracting actionable insights from unstructured text data to support data-driven decision-making.

## Dataset

The dataset used for this analysis is a collection of customer reviews from an e-commerce platform, originally published for a Shopee Code League competition.

* **Original Source:** [Shopee Code League - Sentiment Analysis on Kaggle](https://www.kaggle.com/datasets/davydev/shopee-code-league-20/data)
* **Raw Dataset Link (in this repo):** (https://github.com/alesyasyah/ecommerce-feedback-classifier-ibm-granite/blob/main/datasets/train.csv)

## Tools and Technologies

* **Programming Language:** Python 3 
* **Libraries:** Pandas, Matplotlib, Seaborn, PyTorch, Scikit-learn 
* **AI Framework:** Hugging Face Transformers 
* **Model:** IBM Granite (`ibm-granite/granite-3.0-2b-instruct`) 
* **Environment:** Google Colab (T4 GPU) 

## Analysis Process

The analysis was conducted in a Google Colab notebook. The key steps were:
1.  **Data Preparation:** The raw dataset was loaded, and the 1-5 star ratings were mapped to three sentiment categories: Positive (4-5 stars), Neutral (3 stars), and Negative (1-2 stars). 
2.  **Sentiment Classification:** A zero-shot classification approach was used on a sample of 500 reviews.  [cite_start]The model was given a specific prompt instructing it to classify each review. 
3.  **Evaluation:** The model's predictions were rigorously evaluated by calculating accuracy metrics and generating a confusion matrix to deeply understand its performance and biases. 
4.  **Insight Generation:** Key themes were extracted from the results, focusing not just on what the model got right, but more importantly, on *why* it made mistakes.

## Insights & Findings

The analysis revealed two critical findings regarding the model's real-world performance.

### Finding 1: High Failure Rate & Skewed Distribution
The model's first challenge is reliability. On a sample of 500 reviews, **29% (144 reviews) could not be classified** and resulted in an error, indicating the zero-shot prompt is not robust enough for a large portion of real-world data. 

Among the 356 successfully classified reviews, the model predicted a highly skewed distribution, with **77.5% Positive, 19.7% Negative, and only 2.8% Neutral**. 

![image](https://github.com/user-attachments/assets/473037c4-53f2-4c03-8b70-f354533077dc)


### Finding 2: The "Neutral" Blind Spot
The skewed distribution is explained by the project's most important insight, proven by the confusion matrix: **the model is effectively blind to the 'Neutral' category.**

![image](https://github.com/user-attachments/assets/d0584a5e-8eaa-44ee-8485-0628695ffd87)


As shown above, the model correctly identified only one truly `Neutral` review.  It misclassified the vast majority of neutral reviews as `Positive`.  This means the model has a strong optimistic bias, and the 77.5% positive score is inflated and misleading.

## AI Support Explanation

Artificial Intelligence, specifically the IBM Granite LLM, was central to this analysis.

* **Role of AI:** The AI was used for **Data Classification** to automate the process of reading and categorizing customer reviews. 
* **Methodology:** The technique used was **Zero-Shot Classification via Prompting**. A carefully constructed prompt instructed the model on how to perform the classification task without needing to be retrained. 
* **Model Performance & Limitations:** The model's performance must be viewed in two ways:
    * The **Overall Task Success Rate was 61.0%** across all 500 reviews, factoring in the high failure rate. 
    * However, the **Classification Accuracy was 85.7%** on the reviews it could successfully classify. 
    * The primary limitation discovered was the model's inability to interpret ambiguous, neutral, or mixed-signal feedback, causing it to either fail or misclassify the review as positive.

## Conclusion & Recommendations

### Conclusion
This project successfully demonstrated the use of an advanced AI model for feedback classification. The most valuable outcome was not simply building a classifier, but uncovering the critical limitations and biases of a zero-shot approach on complex, real-world data. This analysis provides a realistic blueprint for leveraging AI while being aware of its pitfalls.

### Recommendations
Based on the finding that the model struggles with ambiguity, the recommendations are strategic rather than operational:

1.  **Change the AI's Task:** Do not use the model for a 3-class sentiment score. Instead, use it as a **2-class classifier** to accurately separate clearly `Positive` reviews from clearly `Negative` ones.
2.  **Create an Intelligent Triage System:** Automatically **flag all ambiguous reviews** (neutral ones and the 29% error cases) for immediate **manual review by a human agent**.
3.  **Evolve the AI's Role:** This transforms the AI from a simple "scorer" into a **smart assistant**. The AI handles the high-volume, easy-to-classify reviews, allowing human experts to focus their valuable time on the nuanced feedback that is crucial for genuine business improvement.
