# Airbnb Price Prediction

## Project Overview
This project aims to predict the price of Airbnb listings using a machine learning approach. The dataset contains information about Airbnb listings, including features like the number of guests, bedrooms, beds, bathrooms, amenities, and location. The project employs Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and a stacking ensemble model to predict listing prices. The final model achieves an R² score of 0.43 and a Mean Absolute Error (MAE) of 4220.05 on the test set.

The project is implemented in Python using a Jupyter Notebook (`Airbnb_Price_Prediction.ipynb`) and leverages libraries like pandas, scikit-learn, TensorFlow/Keras, XGBoost, and Matplotlib/Seaborn for data analysis and modeling.

## Dataset
The dataset (`airbnb.csv`) contains 12,805 Airbnb listings with 23 columns, including:
- **Numerical Features**: `price` (target), `guests`, `bedrooms`, `beds`, `bathrooms`, `amenities_count`, `rating`, `reviews`, `studios`.
- **Categorical Features**: `country`, `features`, `amenities`, `host_name`, `address`, `checkin`, `checkout`.
- **Other Columns**: `id`, `name`, `host_id`, `safety_rules`, `hourse_rules`, `img_links`.

Key characteristics:
- **Missing Values**: `host_name` (0.06%), `checkin` (6.25%), `checkout` (19.13%).
- **Outliers**: Extreme values in `price` (max: 1,907,963), `bathrooms` (max: 50), and `beds` (max: 98).
- **Skewness**: The `price` column is highly right-skewed, requiring outlier handling and potential transformation.

The dataset is not included in this repository due to size constraints but can be sourced or provided separately.

## Project Structure
```
Airbnb_Price_Prediction/
│
├── Airbnb_Price_Prediction.ipynb  # Main Jupyter Notebook with code
├── airbnb.csv                    # Dataset (not included in repo)
├── README.md                     # This file
└── plots/                        # Directory for saved plots (e.g., price_distribution.png)
```

## Methodology
The project follows these steps:

1. **Exploratory Data Analysis (EDA)**:
   - Inspected dataset structure, data types, and missing values.
   - Analyzed distributions of numerical features (`price`, `guests`, etc.) using histograms, violin plots, and boxplots.
   - Identified skewness in `price` and outliers in `bathrooms` and `beds`.

2. **Data Preprocessing**:
   - Parsed the `features` column to extract `guests`, `bedrooms`, `beds`, and `bathrooms`.
   - Handled missing values: Imputed `host_name` with "Unknown", dropped `checkout` due to high missing rate (19.13%).
   - Capped outliers in `price` (e.g., at 99th percentile) and addressed extreme values in `bathrooms` and `beds`.
   - Normalized numerical features using `StandardScaler`.
   - One-hot encoded categorical features like `country`.

3. **Feature Engineering**:
   - Created `amenities_count` by counting amenities in the `amenities` column.
   - Used features: `guests`, `bedrooms`, `beds`, `bathrooms`, `amenities_count`, and one-hot encoded `country`.

4. **Modeling**:
   - **Base Models**:
     - **Artificial Neural Network (ANN)**: Built with Keras, featuring 3 hidden layers (64, 32, 16 neurons), ReLU activation, dropout (30%), and batch normalization. Achieved R²: 0.34, MAE: ~2900 (standardized units).
     - **XGBoost**: Gradient-boosting model, best performer with R²: 0.46.
     - **Random Forest**: Tree-based model, slightly lower performance than XGBoost.
   - **Stacking Ensemble**: Combined predictions from ANN, XGBoost, and Random Forest using a linear regression meta-learner. Final metrics: R²: 0.43, MAE: 4220.05, MSE: 34,126,169.05.

5. **Evaluation**:
   - Evaluated models on a test set (20% of data) using R², MAE, and MSE.
   - Visualized training loss, predicted vs. actual prices, and residuals to assess model performance.

## Results
- **Stacking Ensemble**:
  - **R²**: 0.43 (explains 43% of price variance).
  - **MAE**: 4220.05 (average prediction error in original price units).
  - **MSE**: 34,126,169.05 (indicates large errors for some predictions).
- **Key Insights**:
  - XGBoost outperformed other models (R²: 0.46), suggesting tree-based models are better suited for this tabular dataset.
  - The modest R² and high MAE indicate challenges in capturing price variance, likely due to limited features, price skewness, or unmodeled factors (e.g., listing quality, seasonality).
  - Large MSE suggests significant errors for high-priced listings.

## Installation and Setup
### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `xgboost`, `matplotlib`, `seaborn`

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Airbnb_Price_Prediction.git
   cd Airbnb_Price_Prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not provided, install manually:
   ```bash
   pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn
   ```

3. **Add the Dataset**:
   - Place `airbnb.csv` in the project root directory.
   - Ensure the dataset matches the expected structure (23 columns, as described).

4. **Run the Notebook**:
   ```bash
   jupyter notebook Airbnb_Price_Prediction.ipynb
   ```
   Execute all cells to reproduce the analysis, preprocessing, and modeling.

## Usage
1. Open `Airbnb_Price_Prediction.ipynb` in Jupyter Notebook.
2. Ensure `airbnb.csv` is in the same directory.
3. Run the cells sequentially to:
   - Perform EDA and visualize distributions.
   - Preprocess the data and engineer features.
   - Train and evaluate the stacking ensemble model.
4. Check the `plots/` directory for saved visualizations (e.g., `price_distribution.png`, `predicted_vs_actual.png`).

## Limitations
- **Modest Performance**: The R² (0.43) indicates the model captures only 43% of price variance, likely due to limited features or data noise.
- **High MAE**: Prediction errors (MAE: 4220.05) are significant, especially for diverse listings with a wide price range.
- **Outliers and Skewness**: Extreme values in `price`, `bathrooms`, and `beds` may distort predictions.
- **Missing Features**: Unmodeled factors (e.g., specific amenities, location granularity, seasonality) limit predictive power.

## Future Improvements
- Apply log-transformation to `price` to reduce skewness.
- Parse `amenities` to include specific features (e.g., WiFi, pool) as binary variables.
- Incorporate spatial features (e.g., latitude/longitude) or more granular location data.
- Experiment with advanced models (e.g., deep neural networks, CatBoost) or hyperparameter tuning.
- Address data quality issues (e.g., validate extreme values in `bathrooms`, `beds`).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.


## Contact
For questions or feedback, please contact [punamdash128@example.com] or open an issue on GitHub.

---

*Built with by [PUNAM DAS]*  
*Last Updated: May 12, 2025*
