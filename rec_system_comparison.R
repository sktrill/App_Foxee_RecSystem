library(tidyr, warn.conflicts=FALSE, quietly=TRUE)
library(readr, warn.conflicts=FALSE, quietly=TRUE)
library(dplyr, warn.conflicts=FALSE, quietly=TRUE)


# --------------------------------------------
# Loading
# --------------------------------------------
setwd("D:/Github/App_Foxee_RecSystem")

# load mock CF data 
mock_full <- tbl_df(read.csv("data/mock.csv", stringsAsFactors=FALSE))
 
# # load real CF data from ANN - train set
# train <- tbl_df(read.csv("data/real_train.csv", stringsAsFactors=FALSE))
# 
# # load real CF data from ANN - RMSE test set
# test <- tbl_df(read.csv("data/real_test.csv", stringsAsFactors=FALSE))


# --------------------------------------------
# Recommendation Systems Methods
# --------------------------------------------

# Twitter DIMSUM Notes
# Dimension Independent Matrix Square using MapReduce (DIMSUM)
# by Reza Bosagh Zadeh, Gunnar Carlson
# --------------------------------------------------------------
# Paper link here: http://www.jmlr.org/papers/volume14/bosagh-zadeh13a/bosagh-zadeh13a.pdf
# - brute force methods (collab filtering or matrix factorization)
#   use cosine similarity on all vectors or reduce entire matrix, 
#   these are computationally expensive
# - instead pre-compute a similarity threshold, s, to figure out the vectors that are
#   more worth comparing
# - oversampling parameter, lambda = 4 * log(n) / s
# - algorithm: 
#       1. loop through rows of matrix, traversing columns, c_i, with probability min (1, lambra / ||c_i||
#       2. emit cosine similarity in mapper function (see paper for formula)
#       3. reducer emits random variables whose expectation is the cosine similarity fcn
# - useful for large sparse and skewed matrices



# Collaborative Filtering
# ---------------------------------
# topNN neighbourhood method

# remove user column
mock <- subset(mock_full, select = -user)

# initialize item-item similarity matrix
sim  <- matrix(NA, 
               nrow = ncol(mock),
               ncol = ncol(mock),
               dimnames = list(colnames(mock),colnames(mock))
               )

# convert to matrix for faster column dot product operations than df
mock <- as.matrix(mock)

# loop all column vectors and perform euclidean dot product
for(i in 1:ncol(mock)) {
  for(j in 1:ncol(mock)) {
    # sim[i,j] <- getCos(as.matrix(mock[i]),as.matrix(mock[j]))
    sim[i,j] <- mock[,i] %*% mock[,j] / sqrt(mock[,i] %*% mock[,i] * mock[,j] %*% mock[,j])
  }
}

# convert to dataframe
mock <- as.data.frame(mock)
sim <- as.data.frame(sim)

# number of similar items to extract
TOPITEMS <- 8

# initialize item similarity matrix for top N closest neighbours
items.topN <- matrix(NA, 
                    nrow = ncol(sim),
                    ncol = TOPITEMS,
                    dimnames = list(colnames(sim)))

# loop through all items, order item rows by similarity, take top N rows
for(i in 1:ncol(mock)) 
  items.topN[i,] <- (t(head(n = TOPITEMS,rownames(sim[order(sim[,i], decreasing=  TRUE),][i]))))


# initialize user similarity matrix for 7 closest neighbours
holder <- matrix(NA, 
                 nrow = nrow(mock_full),
                 ncol = ncol(mock_full) - 1,
                 dimnames = list((mock_full$user), colnames(mock_full[-1])))


for(i in 1:nrow(holder)) { # loop users
  for(j in 1:ncol(holder)) { # look items
    # extract user and item name
    user <- rownames(holder)[i]
    item <- colnames(holder)[j]
    
    # remove items that the user has already liked
    if (as.integer(mock_full[mock_full$user == user, item]) == 1)  
      holder[i,j] <- ""
    else {
      # get top N similar items to the item
      topN <- ((head(n = TOPITEMS, (sim[order(sim[, item], decreasing=TRUE),][item]))))
      topN.names <- as.character(rownames(topN))
      topN.sim <- as.numeric(topN[,1])
      
      # drop first column vector (same as item, diagonal line on sim matrix)
      topN.sim <- topN.sim[-1]
      topN.names < topN.names[-1]
      
      # get all users for top N items
      topN.likes <- mock_full[,c("user", topN.names)]
      
      # filter for just the current user (in current row)
      topN.userlikes <- topN.likes[topN.likes$user == user, ]
      topN.userlikes <- as.numeric(topN.userlikes[!(names(topN.userlikes) %in% c("user"))])
      
      # calculate score of items liked by users and top N items similar to item
      holder[i,j] <- sum(topN.userlikes * topN.sim) / sum(topN.sim)
      
    }
  } 
}

users.scores <- holder

# create final recommendation df with item names
finalrecs.method1 <- matrix(NA, 
                            nrow = nrow(users.scores),
                            ncol = 100, 
                            dimnames = list(rownames(users.scores)))
for(i in 1:nrow(users.scores)) 
  finalrecs.method1[i,] <- names(head(n = 100,(users.scores[,order(users.scores[i,], decreasing=TRUE)])[i,]))

# convert to df
finalrecs.method1 <- as.data.frame(finalrecs.method1)

# save file for comparison
write.csv(finalrecs.method1, file="results/method1.csv" , row.names = FALSE)


# Packages
# method2 <- Recommender(getData(e, "train"), "UBCF",param=list(normalize = "Z-score",method="Cosine"))




