from dataset_loader.load import Wikidata5mAliasLoader
from prediction.prediction import LLMPredictor

def main():
    
    data_loader = Wikidata5mAliasLoader()
    dataset = data_loader.get_data()
    predictor = LLMPredictor()
    predictions = predictor.predict_dataset(dataset)

if __name__ == "__main__":
    main()