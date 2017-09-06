%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% =============== Part 1: Loading movie ratings dataset ================
%  You will start by loading the movie ratings dataset to understand the
%  structure of the data.
% 
clear all;

fprintf('Loading movie ratings dataset.\n\n');

%  Load data
% load ('ex8_movies.mat');
% load ('ex8decksusagedata1.mat');
load ('ex8decksusagedata3.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  From the matrix, we can compute statistics like average rating.
% R(1, :) - First row of 0 and 1 for all 943 users
% Y(1, :) - Ratings for Movie 1 for all 943 users
% Y(1, R(1, :)) - Ratings for Movie 1 filter for only ones that are rated
% Y(1, R(0, :)) - Error, 0 not allowed
% fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n',mean(Y(1, R(1, :))));
% fprintf('Average rating for movie 2 (Golden Eye): %f / 5\n\n',mean(Y(2, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Decks');
xlabel('Users');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in 
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
% load ('ex8_movieParams.mat');
% load ('ex8decksparameters.mat');
% load ('ex8decksparameters2.mat');
load ('ex8decksparameters3.mat');

%  Reduce the data set size so that this runs faster
%  num_users = 55; num_features = 8;
num_movies = 45;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);
           
fprintf(['Cost at loaded parameters: %f \n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement 
%  the collaborative filtering gradient function. Specifically, you should 
%  complete the code in cofiCostFunc.m to return the grad argument.
%  
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f \n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%

%  
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
% movieList = loadMovieList();
movieList = loadDeckList();

%  Initialize my ratings
my_ratings = zeros(size(movieList, 1), 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(21) = 1;
my_ratings(28) = 1;
my_ratings(29) = 12;
my_ratings(33) = 1;

% The algorithm should exactly predict 3 for the deck ????

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
% my_ratings(7) = 3;
% my_ratings(32)= 2;
% my_ratings(54) = 4;
% my_ratings(64)= 5;
% my_ratings(66)= 3;
% my_ratings(69) = 5;
% my_ratings(183) = 4;
% my_ratings(226) = 5;
% my_ratings(355)= 5;

fprintf('\n\nNew user Live Pitch count:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
% load('ex8_movies.mat');
% load('ex8decksusagedata1.mat');
load ('ex8decksusagedata3.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
% [Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = size(X, 2);

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
num_users
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
% X - 5 decks x 3 features

Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);
% Theta corresponds to which user ????
% Theta - 944 users x 10 features

fprintf('Learning completed.\n');

fprintf('\ncontinue.\n');
% pause;

%% ================== Part 8: Recommendation for YOU ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

% 1682 movies x 944 users - AVERAGED RATINGS
p = X * Theta';

% 1682 movies x 1 user - MY RATINGS
% Ymean represents average value - normalize or feature scaling
% my_predictions = p(:,1) + Ymean;
my_predictions = p(:, 1);
% my_predictions (my POSSIBLE ratings) - 1682 movies * 1 user (new user)

% movieList - 1682 movies * 1
% movieList = loadMovieList();
movieList = loadDeckList();
% r - values
% ix - index
% [r. ix] = sort ([1, 2, 2, 3, 3, 1], 'descend');
% r = 3   3   2   2   1   1
% ix = 4   5   2   3   1   6
[r, ix] = sort(my_predictions, 'descend');

fprintf('\nTop recommendations for you (User) :\n');
for i=1:size(movieList, 1)
    % j - movie number
    j = ix(i);
    % very likely my_predictions(j) = 5.0 since it is only top 10
    % movieList{j} = name of the movie from original file
    fprintf('Predicting Live Pitch count %.1f for deck id %s\n', my_predictions(j), ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

%% ================ Part 9: Recommendation for similar MOVIE ==================
%  After training the model, you can now make recommendations for similar movies
%  Vectorization: Low Rank Matrix Factorization

movieIndex = 3;
sizeOfSimilarityVector = size(X, 1);
similarity = 100 * ones(sizeOfSimilarityVector, 1);

for i=1:sizeOfSimilarityVector
    if (i != movieIndex)
        similarity(i) = abs(sum(X(movieIndex) - X(i)));
    end
end
[r, ix] = sort(similarity, 'ascend');

fprintf('\nTop 3 SIMILAR decks for %s:\n', movieList{movieIndex});
for i = 1:3
    movieList{ix(i)}
end

%% ================ Part 9: Recommendation for NEW user ==================
%  After training the model, you can now make recommendations for NEW user
%  Vectorization: Mean Normalization
% mean -> sum of ratings / number of ratings -> 1 dimensional vector 
% Y -> Each existing rating - mean value in that row.
% Optimize Theta and X for min cost.
% For user 'j' (ZERO ratings) on movie 'i', (theta' * X(i) + mean(i)) 






