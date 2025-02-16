import joblib
from src import model_path, scaler_path, artifacts_path
from src.train import trainModel
from src.constants import opt_feature

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

class ModelScoring:
    '''
    Class to score the input dataframe
    '''
    def __init__(self, input_df):
        self.input_df = input_df

    def process(self, input_df):
        '''
        This method apply preprocessing on the input dataframe.

        Args:
            input_df: Pandas DataFrame to make predictions on.
        
        Returns;
            input_df_: Input dataframe without nulls.
            input_features: Dataframe with columns required to mak predictions.
        '''
        input_df_ = input_df.dropna()
        input_features = input_df[opt_feature]

        return input_df_, input_features
    
    def predict(self, input_features_array, model):
        '''
        This method applies the trained model to make predictions.

        Args:
            input_features_array: Scaled dataframe.
            model: trained MLP model.

        Returns:
            pred: Numpy array with model predictions.
        '''
        pred = model.predict(input_features_array)

        return pred
    
    def merge_predictions(self, pred, input_df_):
        '''
        This method merges the predictions to the original dataframe

        Args:
            pred: Numpy array with model predictions.
            input_df_: Original input dataframe.
        '''
        df_with_predictions = input_df_.copy()
        df_with_predictions["predicted_diagnosis"] = pred

        return df_with_predictions
    
    def execute(self):
        '''
        Main method that executes the above methods.
        '''
        input_df_, input_features = self.process(self.input_df)
        df_features_scaled= trainModel.standardise_features(input_features, opt_feature, artifacts_path)
        predictions = self.predict(df_features_scaled, model)
        df_with_predictions = self.merge_predictions(predictions, input_df_)

        return df_with_predictions

