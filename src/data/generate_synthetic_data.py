import random
import pandas as pd

categories = ['Category_A', 'Category_B', 'Category_C', 
              'Category_D', 'Category_E', 'Category_F', 'Category_G']

data = {
    'id': [i for i in range(1, 10001)],
    'text': ['This is a sample text for document {}'.format(i) for i in range(1, 10001)],
    'category': [random.choice(categories) for i in range(1, 10001)]
}

df = pd.DataFrame(data)
df.to_csv('data/raw/synthetic_data.csv', index=False)
print('Synthetic data generated and saved to data/raw/synthetic_data.csv')