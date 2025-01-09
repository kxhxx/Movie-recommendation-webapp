import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from datetime import datetime

# File Path
file_path = "/content/drive/MyDrive/ml-20m"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Function to get average ratings
def getAverageRatings(sparseMatrix, axis):
    mean_ratings = {}
    users, movies, ratings = sparse.find(sparseMatrix)
    if axis == 0:  # Average movie ratings
        for movie in np.unique(movies):
            mean_ratings[movie] = ratings[movies == movie].mean()
    elif axis == 1:  # Average user ratings
        for user in np.unique(users):
            mean_ratings[user] = ratings[users == user].mean()
    return mean_ratings

# Function for Sampling Random Movies and Users
def get_sample_sparse_matrix(sparseMatrix, n_users, n_movies, matrix_name):
    np.random.seed(15)  # Ensure reproducibility
    startTime = datetime.now()

    users, movies, ratings = sparse.find(sparseMatrix)
    uniq_users = np.unique(users)
    uniq_movies = np.unique(movies)

    userS = np.random.choice(uniq_users, n_users, replace=False)
    movieS = np.random.choice(uniq_movies, n_movies, replace=False)
    mask = np.logical_and(np.isin(users, userS), np.isin(movies, movieS))

    sparse_sample = sparse.csr_matrix(
        (ratings[mask], (users[mask], movies[mask])),
        shape=(max(userS) + 1, max(movieS) + 1)
    )

    print("Sparse Matrix creation done. Saving it for later use.")
    sparse.save_npz(file_path + "/" + matrix_name, sparse_sample)
    print("Shape of Sparse Sampled Matrix = " + str(sparse_sample.shape))
    print("Time taken: ", datetime.now() - startTime)

    return sparse_sample

# Load Dataset
TrainUISparseData = sparse.load_npz(file_path + "/TrainUISparseData.npz")
TestUISparseData = sparse.load_npz(file_path + "/TestUISparseData.npz")

# Creating Sample Sparse Matrix for Train Data
if not os.path.isfile(file_path + "/TrainUISparseData_Sample.npz"):
    print("Sample sparse matrix is not present in the disk. We are creating it...")
    train_sample_sparse = get_sample_sparse_matrix(TrainUISparseData, 5000, 1000, "TrainUISparseData_Sample.npz")
else:
    print("File is already present in the disk. Loading the file...")
    train_sample_sparse = sparse.load_npz(file_path + "/TrainUISparseData_Sample.npz")
    print("Shape of Train Sample Sparse Matrix = " + str(train_sample_sparse.shape))

# Creating Sample Sparse Matrix for Test Data
if not os.path.isfile(file_path + "/TestUISparseData_Sample.npz"):
    print("Sample sparse matrix is not present in the disk. We are creating it...")
    test_sample_sparse = get_sample_sparse_matrix(TestUISparseData, 2000, 200, "TestUISparseData_Sample.npz")
else:
    print("File is already present in the disk. Loading the file...")
    test_sample_sparse = sparse.load_npz(file_path + "/TestUISparseData_Sample.npz")
    print("Shape of Test Sample Sparse Matrix = " + str(test_sample_sparse.shape))

# Global Average Ratings
globalAvgRating = np.round((train_sample_sparse.sum() / train_sample_sparse.count_nonzero()), 2)
globalAvgMovies = getAverageRatings(train_sample_sparse, axis=0)
globalAvgUsers = getAverageRatings(train_sample_sparse, axis=1)

print("Global average of all movies ratings in Train Set is: ", globalAvgRating)
print("No. of ratings in the train matrix is: ", train_sample_sparse.count_nonzero())

# Function to Create Features for Train Data
def CreateFeaturesForTrainData(SampledSparseData, TrainSampledSparseData):
    startTime = datetime.now()

    sample_users, sample_movies, sample_ratings = sparse.find(SampledSparseData)
    print("No. of rows in the returned dataset: ", len(sample_ratings))

    data = []

    for user, movie, rating in zip(sample_users, sample_movies, sample_ratings):
        row = []

        # User ID, Movie ID, Global Average
        row.append(user)
        row.append(movie)
        row.append(globalAvgRating)

        # User and Movie Averages
        row.append(globalAvgUsers.get(user, globalAvgRating))
        row.append(globalAvgMovies.get(movie, globalAvgRating))

        # Ratings by similar users
        try:
            similar_users = cosine_similarity(TrainSampledSparseData[user], TrainSampledSparseData).ravel()
            similar_users_indices = np.argsort(-similar_users)[1:]
            similar_users_ratings = TrainSampledSparseData[similar_users_indices, movie].toarray().ravel()
            top_similar_user_ratings = list(similar_users_ratings[similar_users_ratings != 0][:5])
            top_similar_user_ratings.extend([globalAvgMovies[movie]] * (5 - len(top_similar_user_ratings)))
            row.extend(top_similar_user_ratings)
        except:
            row.extend([globalAvgRating] * 5)

        # Ratings for similar movies
        try:
            similar_movies = cosine_similarity(TrainSampledSparseData[:, movie].T, TrainSampledSparseData.T).ravel()
            similar_movies_indices = np.argsort(-similar_movies)[1:]
            similar_movies_ratings = TrainSampledSparseData[user, similar_movies_indices].toarray().ravel()
            top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
            top_similar_movie_ratings.extend([globalAvgUsers[user]] * (5 - len(top_similar_movie_ratings)))
            row.extend(top_similar_movie_ratings)
        except:
            row.extend([globalAvgRating] * 5)

        # Append actual rating
        row.append(rating)
        data.append(row)

    print("Total Time: ", datetime.now() - startTime)
    return data

# Generate Features for Train and Test Data
data_rows = CreateFeaturesForTrainData(train_sample_sparse, train_sample_sparse)
test_data_rows = CreateFeaturesForTrainData(test_sample_sparse, train_sample_sparse)

# Save Features to CSV
columns = ["User_ID", "Movie_ID", "Global_Average", "User_Average", "Movie_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "Rating"]
train_regression_data = pd.DataFrame(data_rows, columns=columns)
test_regression_data = pd.DataFrame(test_data_rows, columns=columns)

train_regression_data.to_csv(file_path + "/Training_Data_For_Regression.csv", index=False)
test_regression_data.to_csv(file_path + "/Testing_Data_For_Regression.csv", index=False)

# Load the Data for Validation
train_regression_data = pd.read_csv(file_path + "/Training_Data_For_Regression.csv")
test_regression_data = pd.read_csv(file_path + "/Testing_Data_For_Regression.csv")

print("Train Data Shape: ", train_regression_data.shape)
print("Test Data Shape: ", test_regression_data.shape)

# Surprise Library for Data Structures
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_regression_data[["User_ID", "Movie_ID", "Rating"]], reader)
trainset = data.build_full_trainset()

# Test Data Tuple
testset = list(zip(test_regression_data["User_ID"].values, test_regression_data["Movie_ID"].values, test_regression_data["Rating"].values))

print("Processing Complete.")
