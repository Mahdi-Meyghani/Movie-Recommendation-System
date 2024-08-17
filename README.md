# Movie Recommendation System

This app is a movie recommendation system written in Python, primarily utilizing Jupyter Notebook. It falls under the domain of data science and machine learning and provides users with three different recommendation models:

1. **Popularity-Based Filtering**
2. **Content-Based Filtering**
3. **Collaborative-Based Filtering**

## Features

- Uses data processing libraries like `pandas` and machine learning libraries like `scikit-learn` and `Surprise`.
- Offers three different types of recommendation systems, each with a different algorithm.
- Loads and processes movie data using pandas DataFrame.
- Provides customizable recommendations based on user input.

---

## Algorithms Used

### 1. Popularity-Based Filtering
In the popularity-based filtering approach, the app uses IMDB's weighted rating (WR) formula to rank movies. The weighted rating is calculated as follows:

**IMDB Weighted Rating Formula:**
```
(WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C
```

Where:
- `v` is the number of votes for the movie;
- `m` is the minimum number of votes required to be listed;
- `R` is the average rating of the movie;
- `C` is the mean vote across the dataset.

-  This formula strikes a balance between a movie's average rating and the number of votes it has received. If a movie has a high average rating but very few votes, its weighted rating will be adjusted downward. Similarly, movies with more votes will have their weighted rating shifted closer to the average rating. This method ensures that popular movies with a broad base of reviews get prioritized over highly-rated movies with very few reviews.

The app applies this formula to the dataset loaded through pandas and outputs the top `x` number of movies (default is 10, but this can be customized).

### 2. Content-Based Filtering
This model uses the **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm to compute similarities between movies based on their content, specifically the `overview` column in the dataset.

**TF-IDF Algorithm:**
TF-IDF stands for Term Frequency-Inverse Document Frequency, which is used to convert text into numerical features by assessing how important a word is to a document within a corpus. The term frequency (`TF`) represents how frequently a term appears in a document, while inverse document frequency (`IDF`) measures the significance of the word across all documents.

The `TF-IDF` algorithm computes a score for each word in the document, and the score increases proportionally with the number of times a word appears in a document but is offset by how frequently the word appears across the entire corpus. This helps distinguish relevant terms from common words like "the," "and," etc.

The app applies this algorithm to the overview of movies, using the `scikit-learn` library to calculate the `TF-IDF` matrix. Once the matrix is created, the app computes the similarity between different movies using the **Linear Kernel Function**.

**Linear Kernel Function:**
The linear kernel function is a method used to compute the similarity between two vectors (in this case, the `TF-IDF` vectors of movies). The formula for linear kernel similarity is:
```
K(x, y) = x · y
```
Where:
- `x` and `y` are vectors (e.g., `TF-IDF` vectors of two movies).

- The linear kernel calculates the dot product of two vectors, representing their similarity. For text data like movie overviews, this allows the app to determine how closely related two movies are based on their descriptions. Once the similarity matrix is generated, users can input a movie name and get a list of the most similar movies.

### 3. Collaborative-Based Filtering
The collaborative filtering model implemented here uses **Singular Value Decomposition (SVD)** from the `Surprise` library.

In collaborative filtering, the app makes predictions based on user behavior (ratings), assuming that users who agreed on past movies will likely agree on future ones.

**Collaborative Filtering Process:**
1. Load the dataset with `pandas` DataFrame.
2. Convert the dataframe into a dataset format with `Surprise` library using its `Dataset` class.
3. Prepare a training set for the machine learning model.
4. Apply the SVD model from `Surprise`.
5. Train the model, enabling it to predict how a user will rate a specific movie on a scale of 1 to 5.

- SVD is a matrix factorization technique used to decompose a user-item rating matrix into smaller matrices to identify latent factors that influence both user preferences and movie attributes. It breaks down the large matrix into three smaller ones (U, Sigma, and V), making it easier to compute predictions for new data points. Once trained, the SVD model predicts ratings for movies that a user hasn’t rated, and the app recommends movies based on these predictions.

---

## Installation

### Requirements:
- Python 3.x
- Jupyter Notebook
- Pandas
- Scikit-learn
- Surprise (for collaborative filtering)

### Installation Steps:
1. Clone the repository.
   ```bash
   git clone <https://github.com/Mahdi-Meyghani/Movie-Recommendation-System.git>

2. Navigate to the project directory.
   ```bash
   cd Movie-Recommendation-System

3. Install the required packages.
   ```bash
   pip install -r requirements.txt

4. Run the Jupyter Notebook.
   ```bash
   jupyter notebook

# Usage
### 1. Popularity-Based Filtering:
- The app will automatically rank and show the top 10 most popular movies.
- You can customize the number of movies by modifying the code.

### 2. Content-Based Filtering:
- Enter a movie name, and the app will show the most similar movies based on content.

### 3. Collaborative-Based Filtering:
- Predict user ratings for movies, then recommend movies based on those predictions.

# Data
The dataset used for training and recommendations is loaded through pandas. You can replace the dataset with your own by putting your data instead of `credits.csv`, `movies.csv` and `ratings.csv`.

# Contributing
Feel free to open issues and submit pull requests if you would like to improve the app.

You can copy and paste this directly into your `README.md` file!
