# Results

## 1. Initial Prediction

My prediction before performing calculations: **Rooftop Monsoon**

## 2. Top 3 Without Normalization

| Rank | Song | Similarity Score |
|---|---|---:|
| 1 | Rooftop Monsoon | 0.999994 |
| 2 | Fresher Night | 0.999573 |
| 3 | Canteen Bass | 0.999164 |

## 3. Top 3 With Normalization

| Rank | Song | Similarity Score |
|---|---|---:|
| 1 | Fresher Night | 0.995150 |
| 2 | Canteen Bass | 0.983837 |
| 3 | Hostel Corridor | 0.975356 |

## 4. Comparison

The top-three rankings do not agree.

My initial prediction, Rooftop Monsoon, appears in the top three without normalization but not after normalization.

## 5. Squashing Functions

Input score: **47**

| Function | Output |
|---|---:|
| Sigmoid(47) | 1.000000 |
| ReLU(47) | 47.000000 |
| Tanh(47) | 1.000000 |

Additional tests were performed using inputs -2, -1, 0, 1 and 2.
