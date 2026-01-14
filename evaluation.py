import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


model = tf.keras.models.load_model("cat_dog_classifier.keras")
print("Model loaded successfully.")


test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)


loss, accuracy = model.evaluate(test_set, verbose=1)





y_pred = model.predict(test_set)
y_pred = (y_pred > 0.5).astype(int)

y_true = test_set.classes


print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Cat", "Dog"]))
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")