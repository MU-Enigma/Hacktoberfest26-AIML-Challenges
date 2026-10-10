import math 
import csv
# These are pre made toolboxes that are imported as we require it for mathematical functions like square root etc. and the other gives ability to read excel sheets.

# --- MATH & SQUASHING FUNCTIONS ---

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(v1, v2):
    mag_v1, mag_v2 = magnitude(v1), magnitude(v2)
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag_v1 * mag_v2)

def min_max_normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0.0, float(x))

def tanh(x):
    return math.tanh(x)

# --- DATA PROCESSING ---

def main():
    target_song = [124.0, 210.0, 0.78, 0.82] # this is actually for comparision type. You can edit your comparision to your liking
    dataset = [] # creates empty foler that for reading the songs in the file
    
    try: #reads and analyses the songs.csv file
        with open('datasets/songs.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader) # this skips the first row of the spreadsheet. We need it to analyse the songs and not the function headings
            for row in reader: # looking for one song at a time
                name = row[0]
                features = [float(x) for x in row[1:5]] # for every song it takes, it generates the 4 functions from the song and gives it the necessary values
                dataset.append({"name": name, "features": features})
    except FileNotFoundError: #ensures if you keep the songs.csv file in the directory
        print("Error: datasets/songs.csv not found. Please ensure the file exists or try keeping the file in the correct directory.")
        return

    # Raw Cosine Similarity
    raw_results = []
    for song in dataset:
        sim = cosine_similarity(target_song, song["features"])
        raw_results.append((song["name"], sim))
    
    top_3_raw = sorted(raw_results, key=lambda x: x[1], reverse=True)[:3] #This is for getting the top 3 results before normalization.
    #Do you want Top 5 songs instead? Just change the 3 to 5

    # Min-Max Normalization parameters from dataset (for Leveling the play field)
    min_vals = [min(song["features"][i] for song in dataset) for i in range(4)]
    max_vals = [max(song["features"][i] for song in dataset) for i in range(4)]

    # Normalize target song and dataset
    norm_target = [min_max_normalize(target_song[i], min_vals[i], max_vals[i]) for i in range(4)] #
    
    norm_results = []
    for song in dataset: # so we go throught the library again
        norm_features = [min_max_normalize(song["features"][i], min_vals[i], max_vals[i]) for i in range(4)] # this is where squashing comes in place
        sim = cosine_similarity(norm_target, norm_features)
        norm_results.append((song["name"], sim))

    top_3_norm = sorted(norm_results, key=lambda x: x[1], reverse=True)[:3] #selects the top 3 spotlight songs
 #Do you want top 5 songs? Just change the number to 5
    # Result is displayed here 
    print("Predicted song before running: Hukum") # Change your song here
    print("\nTop 3 Songs (Without Normalization):")
    for name, score in top_3_raw:
        print(f"- {name}: {score:.4f}")
        
    print("\nTop 3 Songs (With Min-Max Normalization):")
    for name, score in top_3_norm:
        print(f"- {name}: {score:.4f}")

    print("\nDo they agree?: Nah, Hukum is dominating") # Feel free to change the observation here!

if __name__ == "__main__": # just tells the script to start executing  the file
    main()
