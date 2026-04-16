import csv

def main():
    with open('data/cleaned_dataset.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cleaned_dataset = list(reader)

    with open('data/cleaned_train_dataset.csv', 'w', encoding='utf-8') as csvfile:
        cleaned_train_dataset = cleaned_dataset[:-1000]
        writer = csv.DictWriter(csvfile, fieldnames=["sentence", "label"])
        writer.writeheader()
        writer.writerows(cleaned_train_dataset)
        print(len(cleaned_train_dataset))

    with open('data/cleaned_test_dataset.csv', 'w', encoding='utf-8') as csvfile:
        cleaned_test_dataset = cleaned_dataset[-1000:]
        writer = csv.DictWriter(csvfile, fieldnames=["sentence", "label"])
        writer.writeheader()
        writer.writerows(cleaned_dataset[-1000:])
        print(len(cleaned_test_dataset))

if __name__ == '__main__':
    main()