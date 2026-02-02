import joblib
import re

def clean_title(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    model = joblib.load("product_category_model.pkl")
    
    print("Product category prediction")
    print("Type 'exit' to quit\n")
    
    while True:
        title = input("Enter product title: ")
        
        if title.lower() == "exit":
            print("Goodbye!")
            break
        
        cleaned = clean_title(title)
        prediction = model.predict([cleaned])
        
        print(f"Predicted category: {prediction[0]}\n")

if __name__ == "__main__":
    main()



python predict_category.py
