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

fprintf('Loading dataset - Number of Meetings wrt focussed Users and Decks\n\n');
pause;

%  Load data
% load ('ex8_movies.mat');
% load ('ex8decksusagedata1.mat');
load ('ex8decksusagedata3.mat');

fprintf('Loading completed\n\n');
pause;

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
xlabel('Users');%

fprintf('\nVisualizing...\n\n');
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

fprintf('Loading Parameters - Users, Decks, View Access params\n\n');
pause;

%  Reduce the data set size so that this runs faster
%  num_users = 55; num_features = 4;
num_movies = 45;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);
           
fprintf(['Use random weights and calculate the error: %f \n'], J);

% fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement 
%  the collaborative filtering gradient function. Specifically, you should 
%  complete the code in cofiCostFunc.m to return the grad argument.
%  
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction;

fprintf('\nCheck relative difference is less than allowable limit\n');
pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f \n'], J);

fprintf('\nCheck relative difference is less than allowable limit\n');
pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%

%  
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction(1.5);

fprintf('\nReady to Train the system...\n');
pause;

%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');
pause;

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
% Y = [my_ratings Y];
% R = [(my_ratings ~= 0) R];

%  Normalize Ratings
% [Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = size(X, 2);

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

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
save ex8theta.mat Theta X;
% Theta corresponds to which user ????
% Theta - 944 users x 10 features

fprintf('Learning completed.\n');

fprintf('\ncontinue.\n');
pause;

%% =========== Part 7.5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
%

lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 8: Recommendation for FIRST user ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

% 1682 movies x 944 users - AVERAGED RATINGS
p = X * Theta';

% 1682 movies x 1 user - MY RATINGS
% Ymean represents average value - normalize or feature scaling
% my_predictions = p(:,1) + Ymean;

% 43rd user that has 12 LP (largest) - Results should be similar to new user example
my_predictions = p(:, 43);
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

fprintf('\nTop THREE recommendations for FIRST user :\n');
for i=1:3
    % j - movie number
    j = ix(i);
    % very likely my_predictions(j) = 5.0 since it is only top 10
    % movieList{j} = name of the movie from original file
    fprintf('Predicting Live Pitch count %.1f for deck name - %s\n', my_predictions(j), ...
            movieList{j});
end
fprintf('\nShow in ClearSlide home page...\n');
pause;

%fprintf('\n\nOriginal ratings provided:\n');
%for i = 1:length(my_ratings)
%    if my_ratings(i) > 0 
%        fprintf('Rated %d for %s\n', my_ratings(i), ...
%                 movieList{i});
%    end
%end

%% ================ Part 9: Recommendation for similar MOVIE ==================
%  After training the model, you can now make recommendations for similar movies
%  Vectorization: Low Rank Matrix Factorization

movieIndex = 29;
sizeOfSimilarityVector = size(X, 1);
similarity = 100 * ones(sizeOfSimilarityVector, 1);

for i=1:sizeOfSimilarityVector
    if (i != movieIndex)
        similarity(i) = abs(sum(X(movieIndex) - X(i)));
    end
end
[r, ix] = sort(similarity, 'ascend');

fprintf('\nCan you do more ?\n');
pause;

fprintf('\nTop THREE SIMILAR decks for deck name = %s:\n', movieList{movieIndex});
pause;

for i = 1:3
    fprintf('\nSimilarity Ranking %d - %s\n', i, movieList{ix(i)});
end
fprintf('\n\n\n');

%% ================ Part 9: Recommendation for NEW user ==================
%  After training the model, you can now make recommendations for NEW user
%  Vectorization: Mean Normalization
% mean -> sum of ratings / number of ratings -> 1 dimensional vector 
% Y -> Each existing rating - mean value in that row.
% Optimize Theta and X for min cost.
% For user 'j' (ZERO ratings) on movie 'i', (theta' * X(i) + mean(i)) 






