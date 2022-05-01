## Install spotifyr package
install.packages('spotifyr')

## Load all necessary packages
library("spotifyr")
library("tidyverse")     
library("rpart")          
library("rpart.plot")     
library("tidymodels")     
library("vip")          
library("psych")
library("reshape2")
library("recommenderlab")
library("RColorBrewer")
library("GGally")
library("performance")

## Load the spotify data set
songs <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv')
View(songs)

####################################### EXPLORATORY ANALYSIS ######################################

## Structure of the data set
str(songs) ## 32833 observations and 23 variables
glimpse(songs)

## Number of songs per genre
songs %>% count(playlist_genre) %>% tibble()

## Order by lowest to highest rating
songs %>% arrange(desc(track_popularity))

## Speechiness across genres
songs %>% group_by(playlist_genre) %>%
  summarize(
    min_speechiness = min(speechiness, na.rm = TRUE),
    max_speechiness = max(speechiness, na.rm = TRUE),
    mean_speechiness = mean(speechiness, na.rm = TRUE)
  ) %>% arrange(desc(mean_speechiness))

## Energy across genres
songs %>% group_by(playlist_genre) %>%
  summarize(
    min_energy = min(energy, na.rm = TRUE),
    max_energy = max(energy, na.rm = TRUE),
    mean_energy = mean(energy, na.rm = TRUE)
  ) %>% arrange(desc(mean_energy))

## Danceability across genres
songs %>% group_by(playlist_genre) %>%
          summarize(
            min_danceability = min(danceability, na.rm = TRUE),
            max_danceability = max(danceability, na.rm = TRUE),
            mean_danceability = mean(danceability, na.rm = TRUE)
) %>% arrange(desc(mean_danceability))

## Valence across genres 
songs %>% group_by(playlist_genre) %>%
  summarize(
    min_valence = min(valence, na.rm = TRUE),
    max_valence = max(valence, na.rm = TRUE),
    mean_valence = mean(valence, na.rm = TRUE)
  ) %>% arrange(desc(mean_valence))

## Tempo across genres
songs %>% group_by(playlist_genre) %>%
  summarize(
    min_tempo = min(tempo, na.rm = TRUE),
    max_tempo = max(tempo, na.rm = TRUE),
    mean_tempo = mean(tempo, na.rm = TRUE)
    ) %>% arrange(desc(mean_tempo))

## Density of audio characteristic per genre
data_hist <- select(songs, 10, 12:23)
data <- melt(data_hist)
  
ggplot(data, aes(x=value)) +
  geom_density(aes(color = playlist_genre),  alpha = 0.5) +
  facet_wrap(facets = vars(variable), ncol=3, scales="free") +
  labs(x='Audio characteristic', y='Density') +
  theme_minimal()

## Correlation between variables
correlation <- cor(songs[, c(12:23)])
View(correlation)

######################################### CLASSIFICATION ########################################

## Make playlist_genre to be last column
songs <- select(songs, 1:9, 11:23, 10)
View(songs)

## Transform the output class to factor
glimpse(songs)

songs <- songs %>%
  mutate(track_id = factor(track_id),
         track_name = factor(track_name),
         track_artist = factor(track_artist),
         track_album_id = factor(track_album_id),
         track_album_name = factor(track_album_name),
         track_album_release_date = factor(track_album_release_date),
         playlist_name = factor(playlist_name),
         playlist_id = factor(playlist_id),
         playlist_genre = factor(playlist_genre),
         playlist_subgenre = factor(playlist_subgenre))

glimpse(songs) ## Verify that the columns are transformed to factor

## Split the data set into training and testing data set
## Train set should contain 80% of the data 
## Test set should contain 20% of the data
set.seed(90)
songs_split <- initial_split(data = songs, prop = 0.8, strata = playlist_genre)
songs_train <- training(songs_split)
songs_test <- testing(songs_split)

## Verifying the split
table(songs$playlist_genre) ## Total per genre
table(songs_train$playlist_genre)
table(songs_test$playlist_genre)

## Verify that the output class is present in both sets
prop.table(table(songs_train$playlist_genre)) * 100
prop.table(table(songs_test$playlist_genre)) * 100

## Create a recipe the audio characteristics
songs_recipe <- recipe(playlist_genre ~ 
                         danceability +
                         energy +
                         loudness +
                         speechiness +
                         acousticness +
                         valence +
                         tempo
                       , data = songs_train)

## Setup the classification tree
setup_decisiontree <- decision_tree(tree_depth = 5, min_n = 50) %>%
  set_mode("classification") %>%
  set_engine("rpart") 

## Defining the workflow for the experiments
workflow_decisiontree <- workflow() %>%
  add_recipe(songs_recipe) %>% 
  add_model(setup_decisiontree)

## Training phase
model_decisiontree <- fit(
  data = songs_train, object = workflow_decisiontree)

## Visualize and plot the decision tree
tree_fit <- model_decisiontree %>% pull_workflow_fit()
rpart.plot(tree_fit$fit, type = 1, extra = 104, box.palette="Blues", roundint = FALSE) 

## Evaluation
multi_metrics <- metric_set(accuracy, precision, recall, f_meas)
model_decisiontree %>% predict(songs_test)

model_decisiontree %>%
  predict(songs_test) %>% 
  bind_cols(songs_test) %>% 
  multi_metrics(truth = playlist_genre, estimate = .pred_class)

#################################### REGRESSION #########################################
## MODELING FOR EXPLANATION
## OLS regression
songs_fit <- linear_reg() %>%
  set_engine("lm") %>%
  fit(track_popularity ~ playlist_genre, data = songs) 

## Check coefficients and intercept
stats::lm(formula = track_popularity ~ playlist_genre, data = songs)

## Augment the model
songs_aug_fit <- augment(songs_fit$fit)

## Model diagnostic
ggplot(songs_aug_fit, aes(x = playlist_genre , y = track_popularity)) +
  geom_point() + theme_minimal() +
  geom_hline(yintercept = 0)

## MODELING FOR PREDICTION
## Train/test data already exists

## Create a recipe 
songs_reg_recipe <- recipe(track_popularity ~ playlist_genre,
                       data = songs_train) %>%
                       step_dummy(all_nominal())


## Estimation
songs_reg_recipe %>% prep() %>% juice()

## Build a model 
songs_reg_model <- linear_reg() %>%
  set_engine('lm')

## Setup a workflow
songs_reg_workflow <- workflow() %>%
  add_model(songs_reg_model) %>%
  add_recipe(songs_reg_recipe)

## Fitting the model using the newly created workflow
songs_reg_fit <- songs_reg_workflow %>%
  fit(data = songs_train)

## Evaluation
songs_pred <- songs_test %>%
  bind_cols(predict(songs_reg_fit, songs_test))

songs_pred %>% metrics(track_popularity, .pred)
