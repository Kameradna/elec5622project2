import os

for root, dir, files in os.walk('data'):
  if root in ['data/validation', 'data/training', 'data/test']:
    print(root)
