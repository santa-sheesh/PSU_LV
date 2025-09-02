file_name = "SMSSpamCollection.txt"

try:
    with open(file_name, 'r', encoding='utf-8') as file:
        ham_words = 0
        spam_words = 0
        ham_count = 0
        spam_count = 0
        spam_exclamation_count = 0
        
        for line in file:
            line = line.strip()
            if line.startswith("ham\t"):
                ham_count += 1
                words = line[4:].split()
                ham_words += len(words)
            elif line.startswith("spam\t"):
                spam_count += 1
                words = line[5:].split()
                spam_words += len(words)
                if line.endswith("!"):
                    spam_exclamation_count += 1

        if ham_count > 0:
            print(f"Prosječan broj riječi u ham porukama: {ham_words / ham_count}")
        if spam_count > 0:
            print(f"Prosječan broj riječi u spam porukama: {spam_words / spam_count}")
            print(f"Broj spam poruka koje završavaju uskličnikom: {spam_exclamation_count}")
except FileNotFoundError:
    print("Datoteka nije pronađena.")
