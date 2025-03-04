# Albatross - An Image Geolocation Model 

NOTE (3/4/25): this project is under continued development!

## Overview

Inspired by the game GeoGuessr, Albatross is an AI designed to predict the coordinates of any given street view panorama from around the world. As is stands, the model's median test error is below 350 miles (think the distance of a straight line from San Francisco to Los Angeles). That said, I am actively experimenting with methods to further mitigate this value (hence, this repo does not yet contain some of the methods in current experimentation/implementation).

## Why This Project?

I loved learning geography growing up and as a consequence, I have spent more hours playing GeoGuessr than I'd like to admit. Especially during the COVID-19 pandemic, GeoGuessr was a way for me to see and explore the world despite spending most of every day at home. With my interest in deep learning expanding in recent months, this felt like the perfect project for me to attempt. Seeing as I have already experimented with my two other all-time favorite games—Minecraft and Chess—an AI to play GeoGuessr was the perfect next step, thematically and in complexity level.

## How It Works

1. **Data** This project uses a dataset of 175000+ street view panoramas from around the world.
2. **Image transformation/partitioning:** Before being fed into the model, the panoramas are cropped to exactly 360 degrees and a 2:1 width/height ratio and then mathematically transformed into four correctly projected 224x224 images of each direction (north, south, east, west). This transformation allowed me to evaluate each of the four images separately, then average their embeddings to achieve more reliable guesses. This also enabled me to more easily test examples outside of the dataset.
3. **Clustering:** For this project, I implemented a classification model based on ResNet18 to first classify images into clusters rather than immediately attempting to accurately predict their coordinates. I came to this decision through observing suboptimal early results and by examining state-of-the-art research into image geoclassification, which suggests that classification modeling is the more optimal approach due to the rather random and nonuniform way land is distributed on Earth. I have experimented with numerous clustering methods, including the Gaussian Mixture Model (allows you to select an exact number of clusters and distributes somewhat evenly), OPTICS (theoretically better than GMM for non-uniformly distributed data but less efficient and more difficult to tune, in my experience), and even utilizing administrative data to create more refined clusters. I am currently working to refine a multi-stage version of the model that begins with broader clusters and then refines according to more precise administrative clusters to achieve higher reliability and precision.
4. **Modeling:** As stated previously, the primary classification model uses ResNet18 and evaluates each directional image individually before averaging their embeddings. I found this approach to achieve more accurate results than passing the panoramas in directly, or by designing a fully custom neural network. I am training using weighted cross-entropy loss (weighted according to the cardinality of each cluster) and an Adam optimizer employing steep L2 regularization.
5. **Coordinate Prediction:** Since by default GeoGuessr evaluates according to geographic error and not cluster classification, the classification outputs must somehow be converted to coordinates. To evaluate the model's results accordingly, I built a class to evaluate the results according to three different methods: simply outputting the centroid of the predicted cluster, outputting the mean of the centroids of the top k clusters, and outputting the median of the centroids of the top k clusters (I also experimented with obtaining the centroid using Kernel Density Estimation but saw suboptimal results).

## Results

Below is a snapshot of the model's current results:

## Reflection

Through this project, I have developed an extreme love-hate relationship with deep learning. While I am very tired of waiting for my code to run, I have definitely gained some insight into the importance of planning training sessions and saving model states along the way, documenting the methods applied and the results obtained, adjusting one thing at once in order to determine what actually affected the results, and going outside every so often (predicting world locations is fun and all, but actually experiencing the world isn't half bad itself).

## Acknowledgements

This project uses the **Street View Panoramas** dataset, which is licensed under the [Google Streetview/Maps license]([[https://www.apache.org/licenses/LICENSE-2.0](https://www.google.com/intl/en-GB_ALL/permissions/geoguidelines/](https://www.google.com/intl/en-GB_ALL/permissions/geoguidelines/))).  

- **Source:** [https://www.kaggle.com/datasets/fratzcan/usa-house-prices/data](https://www.kaggle.com/datasets/nikitricky/streetview-photospheres)
- **Author:** nikitricky

## License

This project is licensed under [Apache 2.0] – see the LICENSE file for details.
