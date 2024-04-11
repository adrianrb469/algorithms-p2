import csv
import random
import string
import time
import matplotlib.pyplot as plt

class LevenshteinDistance:
    def __init__(self):
        pass
    
    def levenshtein_distance_divide_conquer(self, str1, str2):
        if len(str1) == 0:
            return len(str2)  # Caso base: si str1 está vacía, la distancia es la longitud de str2
        elif len(str2) == 0:
            return len(str1) 
        elif str1[-1] == str2[-1]:
            # Si los últimos caracteres de str1 y str2 son iguales, la distancia es la misma que si no los incluyéramos  
            return self.levenshtein_distance_divide_conquer(str1[:-1], str2[:-1])
        else:
            substitution_cost = 1
            substitution = self.levenshtein_distance_divide_conquer(str1[:-1], str2[:-1]) + substitution_cost
            insertion = self.levenshtein_distance_divide_conquer(str1, str2[:-1]) + 1
            deletion = self.levenshtein_distance_divide_conquer(str1[:-1], str2) + 1
            # Necesitamos encontrar el mínimo entre los tres posibles casos
            return min(substitution, insertion, deletion)

    def levenshtein_distance_greedy(self, str1, str2):
        if len(str1) == 0:
            return len(str2) # Caso base: si str1 está vacía, la distancia es la longitud de str2
        elif len(str2) == 0:
            return len(str1)
        
        distance = 0
        i, j = 0, 0
        
        while i < len(str1) and j < len(str2):
            if str1[i] != str2[j]:
                distance += 1 # Incrementar la distancia si los caracteres no coinciden
                if len(str1) > len(str2):
                    i += 1
                elif len(str1) < len(str2):
                    j += 1
                else:
                    # Si las longitudes de las cadenas son iguales, avanzar en ambas cadenas
                    i += 1
                    j += 1
            else:
                # Si los caracteres coinciden, avanzar en ambas cadenas
                i += 1
                j += 1
                
        distance += abs(len(str1) - len(str2))
        
        return distance

    def levenshtein_distance_dynamic(self, str1, str2):
            m = len(str1) + 1
            n = len(str2) + 1
            
            # Crear una matriz de tamaño (m x n) para almacenar los subproblemas
            # La matriz dp almacenará la distancia de Levenshtein entre los prefijos de las cadenas
            dp = [[0] * n for _ in range(m)]
            
            # Inicializar la primera fila y la primera columna de la matriz dp
            # dp[i][0] representa la distancia entre el prefijo de str1 de longitud i y una cadena vacía
            # dp[0][j] representa la distancia entre una cadena vacía y el prefijo de str2 de longitud j
            for i in range(m):
                dp[i][0] = i
            for j in range(n):
                dp[0][j] = j
            
            # Llenar la matriz dp utilizando la subestructura óptima
            for i in range(1, m):
                for j in range(1, n):
                    # Si los caracteres en las posiciones i-1 y j-1 son iguales,
                    # la distancia no se incrementa y se toma el valor de la submatriz superior izquierda
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        # Si los caracteres son diferentes, se toma el mínimo entre tres posibles operaciones:
                        # 1. Insertar un carácter (dp[i][j-1] + 1)
                        # 2. Eliminar un carácter (dp[i-1][j] + 1)
                        # 3. Reemplazar un carácter (dp[i-1][j-1] + 1)
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            
            # La solución al problema original se encuentra en la esquina inferior derecha de la matriz dp
            return dp[m - 1][n - 1]
    
    def time_algorithm(self, algorithm, str1, str2, iterations=10):
        start_time = time.time()
        for _ in range(iterations):
            algorithm(str1, str2)
        end_time = time.time()
        return (end_time - start_time) / iterations

    def generate_execution_time_graph(self, str_pairs, iterations=10):
        algorithms = [
            ("Divide and Conquer", self.levenshtein_distance_divide_conquer),
            ("Greedy", self.levenshtein_distance_greedy),
            ("Dynamic Programming", self.levenshtein_distance_dynamic)
        ]

        execution_times = {algorithm[0]: [] for algorithm in algorithms}
        min_times = {algorithm[0]: float('inf') for algorithm in algorithms}
        max_times = {algorithm[0]: 0 for algorithm in algorithms}

        for str1, str2 in str_pairs:
            for name, algorithm in algorithms:
                time_taken = self.time_algorithm(algorithm, str1, str2, iterations)
                execution_times[name].append(time_taken * 1000)  # Convert to milliseconds
                min_times[name] = min(min_times[name], time_taken * 1000)
                max_times[name] = max(max_times[name], time_taken * 1000)

        plt.figure(figsize=(12, 8))
        for name, times in execution_times.items():
            plt.plot(range(1, len(str_pairs) + 1), times, label=name)

        plt.xlabel('Input Size (Word Pairs)')
        plt.ylabel('Time (milliseconds)')
        plt.title('Comparison of Levenshtein Distance Algorithms')
        plt.legend(loc='upper left')

        # Create a separate box for displaying min and max times
        min_max_text = "Algorithm    Min Time (ms)    Max Time (ms)\n"
        min_max_text += "-" * 50 + "\n"
        
        
        for name in algorithms:
            min_max_text += f"{name[0]:<15} {min_times[name[0]]:.4f}          {max_times[name[0]]:.4f}\n"

        plt.annotate(min_max_text, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()
        
    def save_results_to_csv(self, filename, data):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['String 1', 'String 2', 'Divide and Conquer', 'Greedy', 'Dynamic Programming'])
                writer.writerows(data)
                
    def compute_distance(self, str1, str2):
        divide_conquer_dist = self.levenshtein_distance_divide_conquer(str1, str2)
        greedy_dist = self.levenshtein_distance_greedy(str1, str2)
        dynamic_dist = self.levenshtein_distance_dynamic(str1, str2)
        return divide_conquer_dist, greedy_dist, dynamic_dist

def generate_word_pair(length):
    str1 = ''.join(random.choices(string.ascii_lowercase, k=length))
    str2 = ''.join(random.choices(string.ascii_lowercase, k=length))
    return str1, str2

word_pairs = [
    ("a", "b"),
    ("cat", "dog"),
    ("run", "ran"),
    ("big", "small"),
    ("eat", "ate"),
    ("blue", "red"),
    ("house", "home"),
    ("book", "novel"),
    ("fast", "slow"),
    ("happy", "sad"),
    ("sun", "moon"),
    ("hot", "cold"),
    ("hard", "soft"),
    ("fire", "water"),
    ("tree", "forest"),
    ("north", "south"),
    ("young", "old"),
    ("sharp", "dull"),
    ("summer", "winter"),
    ("sweet", "sour"),
    ("day", "night"),
    ("apple", "banana"),
    ("loud", "quiet"),
    ("wet", "dry"),
    ("clean", "dirty"),
    ("high", "low"),
    ("smooth", "rough"),
    ("simple", "complex"),
    ("light", "darkness"),
    ("happy", "miserable"),
]

# order by length
word_pairs.sort(key=lambda x: len(x[0]))

print(len(word_pairs))

levenshtein = LevenshteinDistance()
levenshtein.generate_execution_time_graph(word_pairs)