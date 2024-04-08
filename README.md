# Collaborative-Filtering-Recommender-Systems
Hello and welcome to our lab for collaborative filtering recommender systems! We'll delve into the world of recommendation engines in this practical exercise and see how collaborative filtering can be used to create a movie recommender system. By the time this lab ends, you will have put important algorithms into practice and learned how to make recommendations that are unique for each user.

Comprehending Recommender Systems: Recommender systems are algorithms that utilize past data to forecast user preferences or offer tailored recommendations. When creating recommender systems, one of the most widely used methods is collaborative filtering.

Examining the Movie Ratings Dataset: For this exercise, we'll make use of a movie ratings dataset that comprises user ratings for a variety of films. The dataset's rows each correspond to a user's rating for a particular film.

Using the Algorithm for Collaborative Filtering Learning:
Now let's explore the algorithm for collaborative filtering learning:

4.1 Teamwork in Filtering Cost Function: The goal of the collaborative filtering cost function is to reduce the squared error between the users' actual ratings and the ratings that are predicted. This is how the cost function is put into practice:
 
# GRADED FUNCTION: cofi_cost_func
# UNQ_C1
def cofi_cost_func(X, W, b, Y, R, lambda_):
“””
Returns the cost for the content-based filtering
Args:
X (ndarray (num_movies,num_features)): matrix of item features
W (ndarray (num_users,num_features)) : matrix of user parameters
b (ndarray (1, num_users) : vector of user parameters
Y (ndarray (num_movies,num_users) : matrix of user ratings of movies
R (ndarray (num_movies,num_users) : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
lambda_ (float): regularization parameter
Returns:
J (float) : Cost
“””
nm, nu = Y.shape
J = 0

# Compute predictions
predictions = np.dot(X, W.T) + b

# Compute squared error
error = (predictions — Y) * R
squared_error = np.sum(error ** 2) / 2

# Regularization term
reg_term = (lambda_ / 2) * (np.sum(W ** 2) + np.sum(X ** 2) + np.sum(b ** 2))

# Total cost
J = squared_error + reg_term

return J
 
Exercise 1: Collaborative Filtering Cost Function Implementation

Learning Movie Recommendations: After the cost function has been implemented, the parameters that minimize the cost function can be learned using optimization techniques like gradient descent.

Creating Recommendations: We are able to forecast ratings for movies that a user has not yet rated once we have learned the parameters. We are able to suggest movies to users according to their tastes thanks to these predictions.
