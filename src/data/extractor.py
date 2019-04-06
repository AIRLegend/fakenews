"""
Módulo para extraer un csv con los tipos de noticias infrarrepresentados

Autor: Álvaro Ibrain
Fecha: 12 de marzo de 2018
"""

import sys
import random
import csv
csv.field_size_limit(99999999999)


def extract(input_data, output_data, types = ['clickbait', 'bias'], offset_col = 3):
    """
    Función para extraer solo los articulos de un tipo del dataset Fake News Corpus

    :param (string) input_data: Path al fichero de lectura
    :param (string) output_data: Path al fichero de escritura
    :param (float) types: Tipos de noticias que se desean seleccionar
    """
    with open(input_data) as file:
        with open(output_data, 'w+') as out:
            header = True
            reader = csv.reader(file)
            writer = csv.writer(out)
            for r in reader:
                if header:
                    #Keep header
                    writer.writerow(r)
                    header = False
                else:
                    #Select only if it is in types
                    try:
                        if r[offset_col] in types:
                            writer.writerow(r)
                    except:
                        pass #If the column is empty

def main():
    inp_file = ""
    out_file = ""

    if len(sys.argv) < 3:
        sys.stderr.write("Error. Argumentos necesarios: <input_file>"+
                " <output_file>")
        return

    types = ['clickbait', 'bias']
    if len(sys.argv) == 4:
        types = sys.argv[3].split(',')

    inp_file = sys.argv[1]
    out_file = sys.argv[2]
    print("Processing...")
    print("Extracting news of type:")
    print(types)
    extract(input_data =inp_file, output_data = out_file, types = types)

if __name__ == '__main__':
    main()
