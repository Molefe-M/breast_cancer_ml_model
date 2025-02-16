# local imports
from src.constants import target_mapping

# third part import
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score


class trainModel:
    '''
    Class object to train the model

    '''
    def __init__(self, path_data, file_name_data, path_model, file_name_model, features, size):
        self.path_data = path_data
        self.file_name_data = file_name_data
        self.features = features
        self.path_model = path_model
        self.file_name_model = file_name_model
        self.size = size

    def load_data(self, path_data, file_name_data):
        '''
        This method loads input data to train the model

        Args:
            path: path where training dataset is saved.
        
        Return:
            df_train: Pandas DataFrame training set.
        '''
        df_train = pd.read_csv(path_data+file_name_data, index_col=False)
        df_train["diagnosis"] = df_train["diagnosis"].map(target_mapping)
        print(f"training set loaded successfully: {df_train.shape}")

        return df_train.dropna()
    
    def features_target_split (self, df_train):
        '''
        Function to split data into target and features
        
        '''
        df_features = df_train.drop('diagnosis', axis = 1)
        target = df_train['diagnosis']
        print(f"Training set split into features: {df_features.shape} and labels{target.shape}")

        return df_features, target
    
    def fit_scaler(self, df_features, features, path_model):
        '''
        This method obtains min-max scaler object for input features.

        Args:
            df_train: Pandas DataFrame training set.
            features: input features List.
            path_model: path to save model artificats.
        
        Return:
            None

        '''
        scaler = MinMaxScaler()
        scaler.fit(df_features[features])
        joblib.dump(scaler, path_model+"scaler.pkl")
        print(f"Scaler saved to the path: {path_model}")

        pass

    @staticmethod
    def standardise_features(df_features, features, path_model):
        '''
        This function standardises the input features using minmax scaler.

        Args:
            df_train:
            features:
            path_model
        '''
        scaler = joblib.load(path_model+"scaler.pkl")
        df_features_scaled = scaler.transform(df_features[features])
        print(f"Scaler loaded from path: {path_model} and applied to the dataset: {df_features_scaled.shape}")

        return df_features_scaled

    def data_split(self, df_features_scaled, target):
        '''
        This method splits data into 80% training and 20% testing (3 marks)
        
        Args: 
            df_features_scaled: DataFrame with scaled features
            target: taget labels
        
        Returns:
            train_feat_array:
            test_feat_array:
            train_target:
            test_target: 
        '''
        features_reduced_np = np.array(df_features_scaled)
        target_np = np.array(target)
        
        train_feat_array, test_feat_array, train_target, test_target = train_test_split(features_reduced_np, target_np, test_size=0.2, random_state=42)
        print(f"Dataset converted to numpy array with train_feat: {train_feat_array.shape},\
              train_target: {train_target.shape}, test_feat: {test_feat_array.shape} and test_target: {test_target.shape}")

        return train_feat_array, test_feat_array, train_target, test_target
    
    def train_model(self, train_feat_array, train_target, path_model, file_name_model, size):
        '''
        This method train the MLP model at given hidden layer and at parameters
        
        Args:
            train_feat_array: 2D Numpy array of training features 
            train_target: target label
            size: hiddel layer size

        Return:
            clf: trained ml classifier
        '''
        clf = MLPClassifier(hidden_layer_sizes=(size,), max_iter=1000, solver='adam', random_state=42)
        clf.fit(train_feat_array, train_target)
        joblib.dump(clf, path_model+file_name_model)
        print(f"Model trained and saved to path: {path_model}")

        pass
    
    def test_model(self, test_feat_array, path_model, file_name_model):
        '''
        This method test the trained model. 

        Args:
            model: trained mlp model
        '''
        model = joblib.load(path_model+file_name_model)
        test_pred_target = model.predict(test_feat_array)
        print(f"Model loaded from path: {path_model} and tested using: {test_feat_array.shape}")

        return test_pred_target
    
    def calculate_metrics(self, test_target, test_pred_target):
        '''
        This method calculates the required metrices.

        Args:

        '''
        precision = precision_score(test_target, test_pred_target, average='weighted')
        recall = recall_score(test_target, test_pred_target, average='weighted')
        f1 = f1_score(test_target, test_pred_target, average='weighted')
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1-score: {:.2f}".format(f1))

        pass
    
    def execute(self):
        '''
        Main method that executes the above methods
        '''
        df_train = self.load_data(self.path_data, self.file_name_data)
        df_features, target = self.features_target_split (df_train)
        self.fit_scaler(df_features, self.features, self.path_model)
        df_features_scaled = self.standardise_features(df_features, self.features, self.path_model)
        train_feat_array, test_feat_array, train_target, test_target = self.data_split(df_features_scaled, target)
        self.train_model(train_feat_array, train_target, self.path_model, self.file_name_model, self.size)
        test_pred_target = self.test_model(test_feat_array, self.path_model, self.file_name_model)
        self.calculate_metrics(test_target, test_pred_target)

        pass