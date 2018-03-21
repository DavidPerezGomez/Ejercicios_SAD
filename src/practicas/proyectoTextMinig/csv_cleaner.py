#!/usr/bin/python

################################################################################
# DISCLAIMER
# Este script solo funciona correctamente asumiendo que la separación entre
# valores de archivo orignial está bien hecha (valueN","valueM), sin espacios
# antes o después de la coma, sin que falten comillas, etc.
# También se confía en que ninuno de los valores contenga la substring
# '","' (comillas dobles, coma, comillas dobles). En caso conrario, se omitirá
# la línea que contenga decho valor.
#
# El objetivo de script es escapar caracteres no permitidos y evitar que haya
# delimitadores de valores mal cerrados al final de las líneas.
################################################################################

import sys


def main():
    try:
        input = sys.argv[1]
        output = sys.argv[2]
        print("input: {}".format(input))
        print("output: {}".format(output))
    except IndexError:
        print("input and output paths required")
        exit(1)

    with open(input, 'r') as csv_file:
        lines = csv_file.readlines()

    # asumimos que la primera línea es la cabecera
    header = lines[0].strip('\n')
    header_fields = separate_values(header)
    num_fields = len(header_fields)
    print("Campos:")
    for field in header_fields:
        print("\t{}".format(field))

    new_lines = []
    for line in lines:
        line = line.strip('\n')
        # si tras quitar el salto de línea el String queda vacío
        # se pasa a analizar la siguiente línea
        # (omitimos líneas vacías)
        if line:
            values = separate_values(line)
            # si el número correcto de valores no es correcto
            # se pasa a analizar la siguiente línea
            if len(values) == num_fields:
                new_values = []
                for value in values:
                    new_values.append(escape_char(value, '\"'))
                    
                new_line = '\",\"'.join(new_values)
                new_line = '\"' + new_line + '\"'
                new_lines.append(new_line)

    with open(output, 'w') as output_file:
        for line in new_lines:
            output_file.write("{}\n".format(line))


def separate_values(line):
    if line.endswith('\"'):
        line = line[:-1]
    if line.startswith('\"'):
        line = line[1:]
    return line.split('\",\"')


def escape_char(line, char):
    new_line = line
    index = new_line.find(char)
    while index != -1:
        new_line = insert_substring(new_line, '\\', index)
        index = new_line.find(char, index+2)
    return new_line


def insert_substring(string, substr, index):
    return string[0:index] + substr + string[index:]


if __name__ == "__main__":
    main()
