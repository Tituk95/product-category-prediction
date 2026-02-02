# Product Category Prediction

## Project Overview
This project focuses on building a machine learning solution for **automatic product category prediction** based solely on product titles.  
The goal is to help an e-commerce platform automatically classify newly added products, reducing manual work, minimizing errors, and improving overall efficiency.

The solution is designed as an end-to-end ML project, covering:
- data exploration and cleaning,
- feature engineering,
- training and evaluation of multiple models,
- saving a trained model for reuse,
- and providing scripts for training and interactive prediction.

---

## Dataset
The dataset (`products.csv`) contains real product listings with the following information:
- Product ID  
- Product Title  
- Merchant ID  
- Category Label (target variable)  
- Product Code  
- Number of Views  
- Merchant Rating  
- Listing Date  

The dataset is located in the `data/` directory.

---

## Project Structure
```md
## Project Structure

product-category-prediction/
│
├── data/
│ └── products.csv
│
├── train_model.py
├── predict_category.py
├── product_category_model.pkl
├── README.md
└── .gitignore
