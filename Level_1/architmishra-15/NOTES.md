**The Problem with a Score of 47**
Probabilities only make sense when they sit between 0 and 1 (or 0% and 100%). You can't have a 4700% chance of liking a song! The **sigmoid** function fixes this by acting like a mathematical compressor. It takes any raw number—no matter how wildly positive or negative—and smoothly squashes it into that perfect 0 to 1 range. Under sigmoid, that awkward 47 just becomes something like 0.99 (a 99% probability).

**Sigmoid vs. Tanh**
Both functions squish numbers into smooth S-shaped curves, but they output different ranges:

* **Sigmoid (0 to 1):** Perfect for straightforward probabilities. Use this when you only care about the exact percentage chance of an event happening (like the odds of adding a song to a playlist).
* **Tanh (-1 to 1):** Ideal when you need to capture opposites with a true "neutral" middle ground. Use this if you want to map a spectrum of feelings: -1 means you aggressively hate the song, 0 means you are completely indifferent, and 1 means you absolutely love it. Sigmoid's center is 0.5, which is less intuitive for representing a neutral stance than a clean 0.
