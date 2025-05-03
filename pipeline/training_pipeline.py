from src.data_processing import DataProcessing
from src.model_training import ModelTraining


if __name__=="__main__":

    data_pipeline = DataProcessing("artifacts/raw/data.csv","artifacts/preprocessed")
    data_pipeline.run()

    pipeline = ModelTraining("artifacts/preprocessed","artifacts/models")
    pipeline.run()
