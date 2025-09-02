lista = []
while True:

    
    print("Unesi broj: ")

    user_input = input()
    if user_input == "Done":
        break
    
    try:
        broj = int(user_input)

        lista.append(broj)

        

    except ValueError:
        
        print("Nije unesen broj")
    
print("Broj brojeva: ",len(lista))
print("Srednja vrijednost: ",sum(lista)/len(lista))
print("Min vrijednost: ",min(lista))
print("Max vrijednost: ",max(lista))

