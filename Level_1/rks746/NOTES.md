# Level 1: Similarity and Squashing

## Introduction
The given dataset songs.csv has 10 songs with 4 parameters each -> tempo_bpm, duration_sec, energy, dancebility.  

The given song has parameter vector - [124, 210, 0.78, 0.82]  
And we have to recommend the most appropriate next song. 

### My prediction
My guess would be Canteen Bass [132, 246, 0.82, 0.85] using maximum-only normalization followed by Manhattan distance. 
1. Divide every feature by its maximum value.
2. Normalize your favourite song using the same maximum values.
3. Calculate the absolute difference between corresponding features.
4. Sum those differences for each song.
5. Recommend the song with the lowest sum (meaning least distance, i.e. closest vector to the input vector)

## Similaity explanation 
Raw consine similarity recommends Rooftop Monsoon, then Fresher Night, then Canteen Bass.  
Cosine similarity with min-max similarity recommends Fresher Night, then Canteen Bass, then Hostel Corridor.

The difference is because cosine similarity is affected by feature magnitudes. 'tempo' and 'duration' dominate the dot product because their magnitudes are much larger than energy and danceability.  
Min-max normalization brings all features to the same range, reducing the dominance of features with larger numerical scales. This allows smaller-scale features to have a more meaningful influence on the similarity calculation

## Squashing explanation 
Sigmoid breaks for high negative values (exp(-x) overflows for sufficiently large negative inputs)  
TanH breaks for high positive values (exp(2*x) overflows for sufficiently large positive inputs)  
Improved implementations avoid it.

## Why is a linear model score of 47 a problem?

A linear model can produce any numerical value, including negative values or values greater than 1. But that wouldn't mean anything. Probability must lie between 0 and 1. Sigmoid solves this by squashing any real-valued input into that range (although it isn't automatically a well-calibrated probability).

## Sigmoid and tanh look similar. When might you prefer one over the other?

Both functions squash large inputs into bounded ranges.  
Sigmoid outputs values between 0 and 1, making it useful when representing probabilities, such as binary classification outputs.  
Tanh outputs values between -1 and 1 and is zero-centred. This can be advantageous in neural networks because its outputs can represent both positive and negative activations, making it useful in hidden layers.
