from typing import NamedTuple
from collections import Counter
import sys
from pprint import pprint
import re
import os
import heapq

MAX_N_GRAMS = 2

class Song(NamedTuple):
    name: str
    artist: str
    album: str
    lyrics: str

class SongClassifier:
    def __init__(self, songs: list[Song]):
        self.songs = songs
        self.songs_features = list() 
        self.songs_probabilities = list()  

        print("Forming features and calculating probabilities...")
        for song in songs:
            features, probabilities = self.form_features(song.lyrics)

            self.songs_features.append(features)
            self.songs_probabilities.append(probabilities)

        print("Sorting features...")
        for i, song in enumerate(self.songs):
            sorted_features_probabilities = sorted(zip(self.songs_features[i], self.songs_probabilities[i]))
            sorted_features, sorted_probabilities = zip(*sorted_features_probabilities)

            self.songs_features[i] = sorted_features
            self.songs_probabilities[i] = sorted_probabilities

        print("Total features:", sum(len(row) for row in self.songs_features))
        print("Optimising features...")

        features = merge_sorted_arrays(self.songs_features)
        print("Total features:", len(features))

        print("Optimising probabilities...")
        probabilities = [[0] * len(features) for _ in range(len(self.songs))]

        # Map old probabilities to new ones
        index_map = {string: index for index, string in enumerate(features)}

        for i, song in enumerate(self.songs):
            for j, feature in enumerate(self.songs_features[i]):
                new_idx = index_map[feature]
                probabilities[i][new_idx] = self.songs_probabilities[i][j]

        self.features = features
        self.probabilities = probabilities

    def predict(self, words: str):
        feature_set, probabilities = self.form_features(words)

        # Map features to existing ones
        index_map = {string: index for index, string in enumerate(self.features)}

        prob_class_and = [1] * len(self.songs)

        for song_id, song in enumerate(self.songs):
            for feature in feature_set:
                index = index_map.get(feature, -1)

                if index != -1:
                    index = index_map[feature]
                    prob_class_and[song_id] *= self.probabilities[song_id][index]

        prob_total_sum = sum(prob_class_and)

        prob_class = [x / prob_total_sum if prob_total_sum != 0 else 0 for x in prob_class_and]

        results = list(sorted(zip(prob_class, self.songs)))
        for idx, (prob, song) in enumerate(results):
            idx = len(self.songs) - idx
            print(f"{idx}. {song.artist} - {song.name}: {prob}")

        best_result = results[len(self.songs) - 1]
        best_song = best_result[1]
        print(best_song.lyrics)



    # Returns features, probabilities
    def form_features(self, words: str) -> tuple[list[str], list[float]]:
        features = re.split(r'[^\w]+', words.strip())
        features = [feature.lower() for feature in features]

        # Generate n-grams
        ngrams = [self.generate_ngrams(features, i) for i in range(MAX_N_GRAMS)]
        
        # Calculate probabilities
        probabilities = list()
        feature_set = list()

        # Append n-grams to current features
        for ngram in ngrams:
            ngram_set = list(set(ngram))

            probabilities.extend(self.calculate_probabilities(ngram_set, ngram))
            feature_set.extend(ngram_set)

        return feature_set, probabilities

    def generate_ngrams(self, words: list[str], n: int) -> list[str]:
        ngrams = zip(*[words[i:] for i in range(n)])
        return [' '.join(ngram) for ngram in ngrams]

    def calculate_probabilities(self, feature_set: list[str], feature_list: list[str]):
        counts = Counter(feature_list)
        probabilities = [counts[feature] / len(feature_list) for feature in feature_set]

        return probabilities



# ----------------------------------------

def form_song(data: str) -> Song:
    lyrics, metadata = re.split(r'\n_{5,}\n', data.strip())
    allowed_keys = ['name', 'artist', 'album', 'lyrics']

    metadata = metadata.splitlines()
    metadata_dict = {}
    for entry in metadata:
        idx = entry.find(' ')

        if idx != -1:
            key = entry[:idx].lower()
            value = entry[idx:].lstrip()

            if key in allowed_keys:
                metadata_dict[key] = value

    song = Song(lyrics=lyrics, **metadata_dict)
    return song



def main(argv: list[str]):
    song_directory = argv[1]
    song_paths = list()
    songs = list()

    print("Adding songs...")
    for root, _, files in os.walk(song_directory):
        for file in files:
            #print("Found song", file)
            path = os.path.join(root, file)
            song_paths.append(path)

    print("Total found songs:", len(song_paths))
    
    print("Forming songs...")
    for i, song_path in enumerate(song_paths):
        try:
            with open(song_path) as f:
                data = f.read()
            song = form_song(data)
            songs.append(song)
            print(f"{i}.", song.artist, song.name)
        except Exception as e:
            print(e, song_path)
            continue

    print("Total found songs:", len(song_paths))
    print('Total formed songs:', len(songs))

    print("Saving songs...")
    with open("song_names.txt", "w") as f:
        f.writelines((f"{song.artist} - {song.name}\n" for song in songs))

    classifier = SongClassifier(songs)

    while True:
        text = input("Enter song lyrics: ")
        classifier.predict(text)


# Ignores duplicates
def merge_sorted_arrays(arrays: list[list]):
    min_heap = []
    last_inserted = None
    
    # Initialize the heap with the first element of each array along with the array index and element index
    for array_index, array in enumerate(arrays):
        if array:  # Ensure the array is not empty
            heapq.heappush(min_heap, (array[0], array_index, 0))
    
    sorted_list = []
    
    while min_heap:
        value, array_index, element_index = heapq.heappop(min_heap)

        if value != last_inserted:
            sorted_list.append(value)
            last_inserted = value
        
        # If there are more elements in the same array, add the next element to the heap
        if element_index + 1 < len(arrays[array_index]):
            next_value = arrays[array_index][element_index + 1]
            heapq.heappush(min_heap, (next_value, array_index, element_index + 1))
    
    return sorted_list


if __name__ == "__main__":
    main(sys.argv)


