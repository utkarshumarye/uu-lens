import numpy as np
from torch.utils.data import DataLoader
from models.cnn_similarity import ResNetSimilarity
from models.autoencoder_similarity import train_autoencoder, load_autoencoder
from models.vision_transformer import VisionTransformerSimilarity
from models.siamese_network import SiameseNetwork, train_siamese_model
from models.deep_hashing import DeepHashingSimilarity, train_deep_hashing
from dataset_loader import load_cifar10
from similarity_search import cosine_similarity

def train_models(train_loader):
    print("Training Autoencoder...")
    train_autoencoder(train_loader)

    print("Training Siamese Network...")
    train_siamese_model(train_loader)

    print("Training Deep Hashing Model...")
    train_deep_hashing(train_loader)

def test_models(models, test_data_loader):
    all_results = []
    for model in models:
        precision_list, recall_list, accuracy_list = [], [], []
        for images, labels in test_data_loader:
            query_image = images[0]
            query_features = model.extract_features(query_image)

            database_features = [model.extract_features(img) for img in images]

            similarities = [cosine_similarity(query_features, db_feat) for db_feat in database_features]
            sorted_indices = np.argsort(similarities)[::-1]

            relevant_indices = [i for i, lbl in enumerate(labels) if lbl == labels[0]]

            true_positives = len(set(sorted_indices[:5]) & set(relevant_indices))
            precision = true_positives / 5
            recall = true_positives / len(relevant_indices)
            accuracy = true_positives / max(len(sorted_indices), len(relevant_indices))

            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)

        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_accuracy = np.mean(accuracy_list)
        all_results.append((model.__class__.__name__, avg_precision, avg_recall, avg_accuracy))

    return all_results

def main():
    train_loader, test_loader = load_cifar10(batch_size=64)
    
    train_models(train_loader)
    
    models = [
        load_autoencoder(),
        ResNetSimilarity(),
        VisionTransformerSimilarity(),
        SiameseNetwork(),
        DeepHashingSimilarity()
    ]
    
    results = test_models(models, test_loader)
    
    print("\nEvaluation Results:")
    print(f"{'Model':<25} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for result in results:
        model_name, precision, recall, accuracy = result
        print(f"{model_name:<25} {precision:<10.2f} {recall:<10.2f} {accuracy:<10.2f}")

    for result in results:
        print(f"{result[0]} - Precision: {result[1]:.2f}, Recall: {result[2]:.2f}, Accuracy: {result[3]:.2f}")

if __name__ == '__main__':
    main()
