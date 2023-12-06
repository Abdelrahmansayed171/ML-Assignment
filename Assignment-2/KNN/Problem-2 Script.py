# %%


class k_nearest_neighbors():

    def __init__(self) -> None:
        self.train=list()

    def __euclidean_distance(self,row_1, row_2):
        distance = 0.0
        for i in range(len(row_1)-1):
            distance += (row_1[i] - row_2[i])**2
        return distance**(1/2) 


    def __neighbors(self,test_row, k):
        distances = list()
        for train_row in self.train:
                dist = self.__euclidean_distance(test_row, train_row)
                distances.append((train_row, dist))
                distances.sort(key=lambda tup: tup[1])
        neighbors_with_distances=distances[:k]
        return neighbors_with_distances   #Tuples of k neighbors and their distances
    

    def predict(self, test_row, k):
        nwd = self.__neighbors(test_row, k) #Neighbors with distance
        total_weighted_votes = {}
        total_weights = 0

        for neighbor, dist in nwd:
            weight = 1 / (dist + 0.000001) #to not divide by 0
            total_weights += weight

            class_label = neighbor[-1]

            if class_label in total_weighted_votes:
                total_weighted_votes[class_label] += weight
            else:
                total_weighted_votes[class_label] = weight

        normalized_votes = {label: weight / total_weights for label, weight in total_weighted_votes.items()} #Weight of every class divided by total weights of both classes
        return max(normalized_votes, key=normalized_votes.get) #Max voted class

    def eval_metric(self,actual, predicted):
        correct = 0

        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct
        
    def fit(self,train, test, num_neighbors):
        self.train=train
        predicted_values = list()
        for row in test:
            output = self.predict(row, num_neighbors)
            predicted_values.append(output)
        actual_values = [row[-1] for row in test]
        corr=self.eval_metric(actual_values,predicted_values)
        print("Model fit successfully using:- \nK: {}\nCorrect predicitons: {}\nTest set instances: {}\nAccuracy: {}%".format(num_neighbors,corr,len(test),str(round(corr / len(test) * 100,2))))
        


# %%
class preprocessing():

    def train_test_split(dataset,ratio):
        
        ratio_idx=round(1-ratio*len(dataset))
        train = dataset[:ratio_idx]
        test = dataset[ratio_idx:]
        return train, test
    
    def normalize(dataset):

        features = [list(map(float, row[:-1])) for row in dataset]
        labels = [row[-1] for row in dataset]

        features_transposed = list(map(list, zip(*features)))

        # Normalization min max
        normalized_features = []
        for feature_values in features_transposed:
            min_value = min(feature_values)
            max_value = max(feature_values)
            normalized_feature = [(float(value) - min_value) / (max_value - min_value) for value in feature_values]
            normalized_features.append(normalized_feature)

        normalized_features = list(map(list, zip(*normalized_features)))

        normalized_dataset = [normalized_feature + [label] for normalized_feature, label in zip(normalized_features, labels)]

        return normalized_dataset

# %%
class pd:
    
    def read_csv(filename,header=True):
        with open(filename, 'r') as f:
            if header:
                next(f)
            results = []
            for line in f:
                line= line.strip()
                words = line.split(',')
                results.append(words)

        return results

# %%
dataset=pd.read_csv('diabetes.csv')

# %%
dataset=preprocessing.normalize(dataset)

# %%
train,test=preprocessing.train_test_split(dataset,0.3)

# %%
knn=k_nearest_neighbors()

# %%
knn.fit(train,test,5)

# %%
for k in [3,6,9,12,14]:
    knn.fit(train,test,k)
    print('─' * 20)


