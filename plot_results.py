import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def process_norm_file(file_path):
    """ Extrae iteraciones y valores de norma residual desde un archivo normR.txt. """
    iterations, dmdc_values, gmres_values = [], [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"Iteración\s+(\d+):\s+Norma Residual DMDc\s*=\s*([\d.eE+-]+),\s+Norma Residual GMRES\s*=\s*([\d.eE+-]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                dmdc_values.append(float(match.group(2)))
                gmres_values.append(float(match.group(3)))
    return iterations, dmdc_values, gmres_values

def plot_norm_file(file_path):
    """ Genera y guarda un gráfico de convergencia de normas residuales de DMDc y GMRES. """
    iterations, dmdc, gmres = process_norm_file(file_path)
    
    if not iterations:
        print(f"Advertencia: No se encontraron datos en {file_path}")
        return

    plt.figure()
    plt.plot(iterations, dmdc, marker='o', linestyle='-', label='DMDc', color='blue')
    plt.plot(iterations, gmres, marker='s', linestyle='--', label='GMRES', color='red')
    plt.xlabel('Iteración')
    plt.ylabel('Norma Residual')
    plt.title(f"Convergencia - {os.path.basename(file_path)}")
    plt.legend()

    # Establecer escala logarítmica en el eje Y
    plt.yscale('log')
    
    output_file = os.path.splitext(file_path)[0] + ".png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_file}")

def process_table_file(file_path):
    """ Extrae valores de norma de la diferencia para distintos (m, p) desde un archivo table.txt. """
    data, m_values, p_values = {}, set(), set()
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"m\s*=\s*(\d+),\s*p\s*=\s*(\d+):\s*Norma de la diferencia\s*=\s*([\d.eE+-]+)", line)
            if match:
                m, p, value = int(match.group(1)), int(match.group(2)), float(match.group(3))
                data[(m, p)] = value
                m_values.add(m)
                p_values.add(p)

    if not data:
        print(f"Advertencia: No se encontraron datos en {file_path}")
        return None, None, None
    
    m_list, p_list = sorted(m_values), sorted(p_values)
    table = np.full((len(m_list), len(p_list)), np.nan)
    
    for i, m in enumerate(m_list):
        for j, p in enumerate(p_list):
            table[i, j] = data.get((m, p), np.nan)
    
    return m_list, p_list, table

def plot_table_file(file_path):
    """ Genera y guarda un gráfico de tabla con valores en función de m y p. """
    m_list, p_list, table = process_table_file(file_path)

    if table is None:
        return

    fig, ax = plt.subplots(figsize=(len(p_list) * 1.5, len(m_list) * 1.5))  # Ajustar tamaño según cantidad de datos
    ax.set_axis_off()  # Ocultar ejes

    # Crear tabla en matplotlib
    table_data = [[f"{table[i, j]:.5f}" for j in range(len(p_list))] for i in range(len(m_list))]
    table_display = ax.table(cellText=table_data,
                             colLabels=[f"p={p}" for p in p_list],
                             rowLabels=[f"m={m}" for m in m_list],
                             cellLoc='center',
                             loc='center')

    # Ajustar tamaño de la tabla para que encaje en la figura
    table_display.auto_set_font_size(False)
    table_display.set_fontsize(10)
    table_display.scale(1.2, 1.2)

    # Guardar la figura
    output_file = os.path.splitext(file_path)[0] + ".png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tabla gráfica guardada: {output_file}")

def main():
    n = 4
    delta = 1.5

    """ Busca y procesa archivos normR.txt y table.txt en la estructura de carpetas test/ZHONGn4/norm y test/ZHONGnX/table """
    base_dirs = glob.glob(f"test/ZHONGorig{n}_delta{delta}/*")

    if not base_dirs:
        print(f"No se encontraron directorios*")
        return

    norm_dir, table_dir = base_dirs

    norm_files = glob.glob(os.path.join(norm_dir, "*.txt"))
    for nf in norm_files:
            plot_norm_file(nf)

    table_files = glob.glob(os.path.join(table_dir, "*.txt"))
    for tf in table_files:
            plot_table_file(tf)

if __name__ == '__main__':
    main()
