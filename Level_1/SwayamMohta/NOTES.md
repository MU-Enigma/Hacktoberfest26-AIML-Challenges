# Level 1 Notes: Squashing and Probabilities

### Why can't a score of 47 be a probability, and how does sigmoid fix it?
A probability is a chance between 0% and 100% (or 0.0 and 1.0). A raw model score like 47 cannot be a probability because it is far above 1.0—you cannot be 4,700% sure you will like a song!

Sigmoid fixes this by acting like a smooth compressor. It takes any score from negative to positive infinity and squeezes it strictly between 0 and 1. When given 47, sigmoid outputs approximately 1.0 (99.999...%), meaning the model is virtually certain you will love the song.

### Sigmoid or Tanh: When would you prefer one over the other?
Choose **sigmoid** when you want an explicit probability or a yes/no decision (e.g., "will you listen to this song?"), because its output lies between 0 and 1.

Choose **tanh** when your data has a natural positive-versus-negative sentiment centered at zero (e.g., rating feedback from -1 for "dislike", 0 for "neutral", to +1 for "like"). Tanh maps inputs to (-1, 1), preserving the direction and sign of the feedback.
