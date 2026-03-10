import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def process_norm_file(file_path):
    """Extrae iteraciones y valores de norma residual desde un archivo normR.txt con DMDc y GMRES."""
    iterations, dmdc_values, gmres_values = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"Iteración\s+(\d+):\s+Norma Residual DMDc\s*=\s*([\d.eE+-]+),\s+Norma Residual GMRES\s*=\s*([\d.eE+-]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                dmdc_values.append(float(match.group(2)))
                gmres_values.append(float(match.group(3)))
    return iterations, dmdc_values, gmres_values

def process_tau_norm_file(file_path):
    """Extrae iteraciones y valores de norma residual para DMDc tau desde un archivo que contiene 'tau_values'."""
    iterations, tau_values = [], []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"Iteración\s+(\d+):\s+Norma Residual DMDctau\s*=\s*([\d.eE+-]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                tau_values.append(float(match.group(2)))
    return iterations, tau_values

def plot_combined_norm_files(file_path):
    """
    Genera y guarda un gráfico de convergencia que incluye:
      - DMDc (línea azul) y GMRES (línea roja) extraídos de file_path,
      - DMDc tau (línea verde) si existe el archivo correspondiente.
    """
    # Procesa el archivo principal normR.txt
    iterations, dmdc, gmres = process_norm_file(file_path)
    if not iterations:
        print(f"Advertencia: No se encontraron datos en {file_path}")
        return

    # Genera el nombre del archivo tau correspondiente reemplazando "normR" por "normR_tau_values"
    base = os.path.basename(file_path)
    tau_file_name = base.replace("normR", "normR_tau_values")
    tau_file_path = os.path.join(os.path.dirname(file_path), tau_file_name)

    tau_iterations, dmdctau = [], []
    if os.path.exists(tau_file_path):
        tau_iterations, dmdctau = process_tau_norm_file(tau_file_path)
    else:
        print(f"No se encontró el archivo tau: {tau_file_path}")

    plt.figure()
    plt.plot(iterations, dmdc, marker='o', linestyle='-', label='DMDc', color='blue')
    plt.plot(iterations, gmres, marker='s', linestyle='--', label='GMRES', color='red')
    if tau_iterations and dmdctau:
        plt.scatter(tau_iterations, dmdctau, linestyle='-.', label='DMDc tau', color='green')
        #plt.plot(tau_iterations, dmdctau, marker='^', linestyle='-.', label='DMDc tau', color='green')
    
    plt.xlabel('Iteración')
    plt.ylabel('Norma Residual')
    plt.title(f"Convergencia - {os.path.splitext(base)[0]}")
    plt.legend()
    plt.yscale('log')
    
    output_file = os.path.splitext(file_path)[0] + ".png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_file}")

def process_table_file(file_path):
    """Extrae valores de norma de la diferencia para distintos (m, p) desde un archivo table.txt."""
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
    """Genera y guarda un gráfico tipo tabla con los valores en función de m y p."""
    m_list, p_list, table = process_table_file(file_path)
    if table is None:
        return

    fig, ax = plt.subplots(figsize=(len(p_list) * 1.5, len(m_list) * 1.5))
    ax.set_axis_off()

    table_data = [[f"{table[i, j]:.5f}" for j in range(len(p_list))] for i in range(len(m_list))]
    table_display = ax.table(cellText=table_data,
                             colLabels=[f"p={p}" for p in p_list],
                             rowLabels=[f"m={m}" for m in m_list],
                             cellLoc='center',
                             loc='center')
    table_display.auto_set_font_size(False)
    table_display.set_fontsize(10)
    table_display.scale(1.2, 1.2)

    output_file = os.path.splitext(file_path)[0] + ".png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tabla gráfica guardada: {output_file}")

def main():
    # Valores de n y delta (para la misma simulación)
    n = 4
    delta = 1.7
    base_path = f"test/ZHONGorig{n}_delta{delta}"

    # Se asume que dentro de base_path existen dos carpetas: 'norm' y 'table'
    norm_dir = os.path.join(base_path, "norm")
    table_dir = os.path.join(base_path, "table")

    if not os.path.exists(norm_dir):
        print("Directorio 'norm' no encontrado")
        return
    if not os.path.exists(table_dir):
        print("Directorio 'table' no encontrado")
        return

    # Procesa todos los archivos normR.txt (excluyendo los que contienen "tau_values")
    norm_files = glob.glob(os.path.join(norm_dir, "*.txt"))
    for nf in norm_files:
        if "tau_values" not in os.path.basename(nf):
            plot_combined_norm_files(nf)

    # Procesa todos los archivos table.txt
    table_files = glob.glob(os.path.join(table_dir, "*.txt"))
    for tf in table_files:
        plot_table_file(tf)

if __name__ == '__main__':
    main()
