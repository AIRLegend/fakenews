"""
Módulo para procesar un dataset grande (varios GB) con pocos recursos de memoria.

Autor: Álvaro Ibrain
Fecha: 4 de marzo de 2018
"""

import sys
import random
import csv
csv.field_size_limit(99999999999)


def sample_file(input_data, output_data, percent = 0.2):
    """
    Función para extraer muestras aleatorias de un csv grande. Sampleando de esta
    manera sólo se carga en memoria una fila cada vez, lo que permite que sin muchos
    recursos se pueda procesar un dataser csv muy grande.

    :param (string) input_data: Path al fichero de lectura
    :param (string) output_data: Path al fichero de escritura
    :param (float) percent: Porcentaje del fichero a analizar. Número entre (0, 1]
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
                    if random.random() < prob:
                        writer.writerow(r)


def main():
    inp_file = ""
    out_file = ""
    percentage = -1

    if len(sys.argv) < 3:
        sys.stderr.write("Error. Argumentos necesarios: <input_file>"+
                " <output_file> [percentage] ")
        return
    if len(sys.argv) <= 3:
        inp_file = sys.argv[1]
        out_file = sys.argv[2]
        sample_file(inp_file, out_file)

    if len(sys.argv) == 4:
        percentage = sys.argv[3]
        sample_file(inp_file, out_file, percent = percentage)


if __name__ == '__main__':
    main()
