import os 
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,input_path, output_path):
        self.input_path =input_path
        self.output_path = output_path
        self.model= xgb.XGBClassifier()
        self.x_train= None
        self.x_test =None
        self.y_test=None
        self.y_train= None

        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Model Training Initialized...")

    def load_data(self):
        try:
            self.x_train =joblib.load(os.path.join(self.input_path, "x_train.pkl"))
            self.x_test =joblib.load(os.path.join(self.input_path, "x_test.pkl"))
            self.y_train =joblib.load(os.path.join(self.input_path, "y_train.pkl"))
            self.y_test =joblib.load(os.path.join(self.input_path, "y_test.pkl"))

            logger.info("preprocessed Data Loaded for model training successfully..")

        except Exception as e:
            logger.error(f"Error while loading the data: {e}")
            raise CustomException("Failed to load the data", e)
        
    def train_model(self):
        try:
            self.model.fit(self.x_train, self.y_train)

            joblib.dump(self.model, os.path.join(self.output_path, "model.pkl"))

            logger.info(" Model Trained and saved Successfully...")

        except Exception as e:
            logger.error(f"Error while Training model {e}")
            raise CustomException("Failed to train model", e)
        
    def eval_model(self):
        try:
            training_score = self.model.score(self.x_train,self.y_train)
            logger.info(f" Trained model score {training_score}")

            y_pred = self.model.predict(self.x_test)
            
            accuracy= accuracy_score(self.y_test, y_pred)
            precision= precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1= f1_score(self.y_test,y_pred, average="weighted")
            
            logger.info(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1_score: {f1}")

            logger.info(f"Model Evaluation completed...")
        except Exception as e:
            logger.error(f"Error while Evaluating model: {e}")
            raise CustomException("Failed to train the model", e)
        
    def run(self):
        self.load_data()
        self.train_model()
        self.eval_model()

        logger.info("Model Training pipeline completed...")


if __name__=="__main__":

    input_path = "artifacts/preprocessed"
    output_path = "artifacts/models"

    pipeline = ModelTraining(input_path,output_path)
    pipeline.run()
