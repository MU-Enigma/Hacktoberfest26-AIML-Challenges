## level 1

 well , Rooftop Monsoon, as its tempo (124) matches mine exactly and its duration is almost identical to mine (210). I matched on these two and gave less weight to energy and danceability.

## Results

```
 Raw cosine similarity
1 Rooftop Monsoon 0.999993619089402
2 Fresher Night 0.9995730769230412
3 Canteen Bass 0.9991635821265635
4 Hostel Corridor 0.9951983828084378
5 Midnight Metro 0.9933736354357433
6 Chai Break Acoustic 0.993286373604171
7 Bus Pass Anthem 0.9928274797495152
8 Exam Week Lofi 0.9909814417713058
9 Sprint to Class 0.9725906918402327
10 Last Bench Ballad 0.9723163025533855
Min-max normalized cosine similarity
1 Fresher Night 0.9951504293354171
2 Canteen Bass 0.9838366728943881
3 Hostel Corridor 0.9753556838190839
4 Bus Pass Anthem 0.9157721066532208
5 Sprint to Class 0.8856411244359877
6 Midnight Metro 0.8049269693274488
7 Chai Break Acoustic 0.741264810425164
8 Exam Week Lofi 0.7146002952234312
9 Rooftop Monsoon 0.6423666248766786
10 Last Bench Ballad 0.33671311229536516
```

Well now to answer some questions:-

**>>> **q = [124, 210, 0.78, 0.82]

**>>> **(0.78**2 + 0.82**2) / (124**2 + 210**2 + 0.78**2 + 0.82**2) * 100

**0.00215342729656195**

**>>> **124 / 210

**0.5904761904761905**

**>>> **from similarity import load_songs, DATA_PATH

**>>> **titles, names, rows = load_songs(DATA_PATH)

**>>> **for t, r in zip(titles, rows):

**... **    **print(r[0] / r[1], t)**

**...** **** **   **

**0.5933014354066986 Rooftop Monsoon**

**0.5365853658536586 Canteen Bass**

**0.42162162162162165 Exam Week Lofi**

**0.5517241379310345 Fresher Night**

**0.30666666666666664 Last Bench Ballad**

**0.7653061224489796 Bus Pass Anthem**

**0.4435483870967742 Chai Break Acoustic**

**0.46511627906976744 Hostel Corridor**

**0.9659090909090909 Sprint to Class**

**0.4444444444444444 Midnight Metro**

**>>>** ****

so this is the output form python shell

## Why they differ
the share of my song from energy and danceabiliry is like 0.00215% that is as the vectors of these two are really small thereupon contribuing less to the direction , the large mag of tempo + duration allows me to say since tempo and duration dominate the vector's length, the dirn is basically set by tempo and duration alone, so comparing directions is really just comparing tempo/duration ratios

however when we normalise it we actually see the contributions and that can be reflected in the change

Rooftop Monsoon ranks first in raw cosine because its tempo/duration ratio is 0.593 which is the closest of all 10 songs to my own ratio of 0.59

My prediction was Rooftop Monsoon. It is rank 1 in the raw list, which matches my prediction, but rank 9 in the other list which is simply sad but yep.. it didnt match my inital hypotheisis

I trust the raw ranking more, because a song doesn't have to feel exactly the same to be a good foloow up tempo and overall pacing can matter more than energy and danceability. Even though Rooftop Monsoon's energy and danceability are much lower than mine, its tempo and duration are nearly identical to mine which is what matters to me


## Breaking sigmoid

sigmoid_naive(-710) crashes with overflow error. A Python float can only hold some range and Python raises an error instead of rounding it wrong.

To fix it, for negative x, I multiplied the top and bottom of the formula by e^x. This gives the same answer, but now the exponent going into exp() is x itself, which is negative here, so e^x is a small number between 0 and 1 — it can never be too big to store.

sigmoid(-710) now doesnt crash 
