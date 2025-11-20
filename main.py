import os
import sys

def main():
    while True:
        print("\n--- MENU PROGETTO ESAME GTSRB ---")
        print("1. Scarica dati e mostra statistiche (Analisi Dataset)")
        print("2. Addestra Modello (Train)")
        print("3. Valuta Modello (Matrice Confusione)")
        print("4. Demo: Predici immagine casuale")
        print("0. Esci")
        
        choice = input("Scegli un'opzione: ")

        if choice == '1':
            os.system(f"{sys.executable} src/data_setup.py")
        elif choice == '2':
            print("NOTA: Per cambiare modello (Custom/ResNet), modifica la variabile MODEL_TYPE in src/train.py")
            os.system(f"{sys.executable} src/train.py")
        elif choice == '3':
            os.system(f"{sys.executable} src/evaluate_detailed.py")
        elif choice == '4':
            os.system(f"{sys.executable} src/predict_demo.py")
        elif choice == '0':
            break
        else:
            print("Scelta non valida.")

if __name__ == "__main__":
    main()